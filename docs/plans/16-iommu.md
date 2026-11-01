# AMD IOMMU (AMD-Vi)

`MachineConfig::iommu(true)` adds an AMD IOMMU as PCI function 00:00.2 and puts every PCI function's DMA behind it. Code is in `crates/uemulator/src/dev/iommu/`. It exists for Windows 11 and Linux, which use it for DMA protection and (Windows with VBS, Linux with x2APIC and large guests) interrupt remapping. They find it through the IVRS table that `acpi/` publishes (`MachineConfig::acpi`, on by default; CLI `--iommu`), and a real Debian 6.12 kernel drives it: see "Verified in a guest" below.

Spec: AMD I/O Virtualization Technology (IOMMU) Specification, rev 3.x (publication 48882). QEMU reference: `hw/i386/amd_iommu.{c,h}` (register layout, command set, event encoding, interrupt remapping decision tree), read for behaviour and not transliterated. Consumer used as the checklist of what must work: Linux `drivers/iommu/amd/amd_iommu_types.h` (bit positions, the IOMMU page table encoding `PAGE_SIZE_PTE`, the event flags).

## Layout

| file | contents |
|------|----------|
| `iommu/mod.rs` | `Iommu` state, the PCI function (config space, capability block, MSI), the MMIO register file, `translate` / `probe` |
| `iommu/commands.rs` | command buffer execution |
| `iommu/events.rs` | event encoding, the log, overflow, the event interrupt |
| `iommu/tables.rs` | device table entries, the 1 to 6 level page walk |
| `iommu/irq.rs` | `remap_interrupt`, `ioapic_request`, IRTE decoding |
| `iommu/ivrs.rs` | `Iommu::ivrs_table` |
| `iommu/tests.rs` | the fake driver and 76 tests |

## PCI function and registers

- 00:00.2, vendor 0x1022 device 0x1419, class 08h/06h. The host bridge at 00:00.0 turns on the multifunction bit when the IOMMU is present; Linux and Windows only scan functions 1-7 of a slot when function 0 says so.
- Capability 0x40 (id 0Fh, type 3, EFRSup and CapExt set): base address low/high with an enable bit (preset to 0xFD200000 enabled, because nothing programs it; the guest may move or disable it), range register (RngValid clear: the IVHD says which devices), misc info (48-bit PA, 64-bit VA). MSI capability at 0x58 (64-bit, one message) delivers the event and completion-wait interrupts: `Iommu::take_msi()` returns the message once, or `None` if the guest has MSI off. Delivering it is the LAPIC's job.
- MMIO window 16 KiB at the capability's base: device table base (size field = pages-1), command buffer and event log bases (length 2^8..2^15 entries, writing a base resets its head and tail), control, exclusion range, extended feature register, head/tail pointers, status (overflow, event interrupt and completion-wait interrupt are write-1-to-clear; the running bits mirror CmdBufEn and EventLogEn). 32-bit halves are accessible. PPR, GA log, SMI filters, hardware error and performance counter registers read zero and ignore writes.
- Extended features advertised: PreFSup, XTSup (x2APIC), IASup, GASup (128-bit IRTEs), HATS 6-level. Not advertised: PPR, guest translation (GT), NX, HE, performance counters, ATS. XTSup brings control bit XTEn (the top byte of an IRTE's destination lives in bits 23:16 of its high qword) and IntCapXTEn: with it the event and completion-wait interrupts are not MSIs but the vector and 32-bit destination software wrote to the XT general interrupt register (0x170), sent straight to the LAPIC without remapping (`Iommu::take_interrupt`). The PPR (0x178) and GA log (0x180) registers are stored only.

## Commands, events

- Executed synchronously when software writes the tail or sets CmdBufEn: COMPLETION_WAIT (store and interrupt), INVALIDATE_DEVTAB_ENTRY, INVALIDATE_IOMMU_PAGES (drops the whole domain: over-invalidating is legal), INVALIDATE_IOTLB_PAGES, INVALIDATE_INTERRUPT_TABLE (no-op, IRTEs are read on every use), PREFETCH, COMPLETE_PPR_REQUEST, INVALIDATE_IOMMU_ALL. An unknown opcode logs ILLEGAL_COMMAND_ERROR with the command's address and stalls the buffer (CmdBufRun clear, head left on it) until software toggles CmdBufEn; a completion-wait store to unmapped memory does the same with COMMAND_HW_ERROR.
- Events: ILLEGAL_DEV_TABLE_ENTRY, IO_PAGE_FAULT (flags PR/RW/I as Linux reads them), DEV_TAB_HW_ERROR, PAGE_TAB_HW_ERROR, ILLEGAL_COMMAND_ERROR, COMMAND_HW_ERROR. Faults found during DMA are queued (`GuestMem::read` is `&self`) and written to the log by `service`, which every MMIO write and the PCI poll call. A full log sets EventOverflow and stops (EventLogRun clear) until software toggles EventLogEn, as `amd_iommu_restart_event_logging` does. Events while the log is off are dropped.

## Translation

- `Iommu::translate(mem, requester, iova, access) -> Translation { phys, bytes_left }`: IOMMU off or no device table programmed = identity. Order: IOTLB hit, DTE fetch, V=0 passes through (QEMU; Linux blocks with V=1, TV=1, mode 0, no IR/IW), reserved DTE bits are an illegal entry, the exclusion range (all devices with ExclusionAllow, else those with DTE.EX), TV=0 passes through, mode 0 passes through if the DTE's IR/IW allow the access, else the walk.
- Walk: modes 1..6, permissions are the AND of the DTE's IR/IW and every entry's, NextLevel may skip levels, 0 is a leaf of the level's size (4 KiB, 2 MiB, 1 GiB...), 7 is a leaf whose size is 2^(p+1) for the lowest clear address bit p >= 12. Such an entry must be repeated in every slot the page covers; the walk does not search for the first. An address above what the mode covers is not present.
- IOTLB: 4 KiB granular, keyed by (requester, page), carries domain and permissions, dropped by the invalidation commands, by changing the device table base or the IOMMU enable, and wholesale at 1024 entries. A translation stays in use until the guest invalidates it, as on hardware.
- Interrupt window 0xFEE00000..0xFEEFFFFF: DMA there is refused without an event (interrupt messages go through `remap_interrupt`, not memory).

## DMA hook

`GuestMem::with_dma(&mut iommu, requester, |mem| ..)` makes every `read`/`write`/`contains` inside the closure a device access from `requester`, split at translation boundaries; `read_raw`/`write_raw` and the `PhysRead`/`PhysWrite` traits are the untranslated physical view (the IOMMU's own table reads, the event log). The IOMMU is moved into the `GuestMem` for the closure and put back after, so no lifetime ties them. `contains` probes without permissions (a write-only buffer still passes) and logs the not-present fault when it refuses, since the caller is about to fail the device's request. Hooks in shared code, all marked by `iommu`: `PciHost::with_iommu`, `PciHost::{iommu_mmio_read, iommu_mmio_write}` called first in `Devices::mmio_read/write`, `PciHost::mmio_write` and `poll` wrap each virtio function in `with_dma` with requester = its devfn, `PciHost::host_bridge` sets the multifunction bit, `DevConfig::iommu`, `MachineConfig::iommu`.

## Interrupt remapping

`Iommu::remap_interrupt(mem, InterruptRequest { requester, address, data }) -> Result<RemappedInterrupt, InterruptFault>` is what `Devices::deliver` calls for every device interrupt; an IOAPIC redirection entry becomes a request with `ioapic_request_raw` (requester = the IOAPIC's IVRS special-device id; with remapping on its vector field carries the IRTE index and its delivery-mode bits the index's top three bits, so reserved mode values must not be rejected). IOMMU off or IV=0: the message is returned as sent. V is not consulted: Linux leaves V clear in the I/O APIC's and HPET's entries and still expects remapping. IV=1: IntCtl abort / forward / remap for fixed and lowest-priority types (SMI is unsupported), NMI/INIT/ExtInt only with their DTE pass bit (LINT1 pass also admits NMI and LINT0 pass ExtInt, my reading of the spec's pin semantics). Remap reads the IRTE at `table + index * 4` (or `* 16` with control.GAEn), index = data[10:0], bounded by 2^IntTabLen (1..11); RemapEn clear aborts (SuppIOPF silences the event), guest mode and non-fixed/arbitrated types are unsupported. Aborts log an IO_PAGE_FAULT with the interrupt flag and the message address.

## IVRS

`Iommu::ivrs_table(&IvrsTopology { pci_devices, ioapics, hpets })` returns the complete table, checksum included: IVinfo (EFRSup, 48-bit PA, 64-bit VA), then IVHD blocks of types 10h, 11h and 40h with the IOMMU's device id and capability offset, its MMIO base, the EFR image (11h/40h), SELECT device entries, and special entries for IOAPICs and HPETs. ACPI generation puts it in the XSDT; the requester ids of the IOAPIC and HPET must match what the interrupt controllers pass to `remap_interrupt`.

## Integration

- `dev/mod.rs` `Devices::deliver` is the one place interrupts meet the IOMMU. Sources: I/O APIC pins (requester `IOAPIC_REQUESTER` = 00:14.0, which Linux insists on as the "southbridge" I/O APIC id, else it disables remapping), the HPET's FSB messages (`HPET_REQUESTER` = 00:14.1), virtio MSI-X messages (requester = the function's devfn) and the IOMMU's own MSI (its devid). With the IOMMU off (or no device table) the message goes to the LAPIC as decoded; with it on, `remap_interrupt` decides. Remapping reads the IRTE from guest RAM, which the CPU's interrupt-polling path does not have, so sources raised there wait in `Devices::deferred` and are delivered by the next `tick` or MMIO write (at most 4096 instructions later). Level-triggered I/O APIC pins remember the vector the LAPIC actually received (`IoApic::note_delivered`) because the EOI carries it, not the index in the redirection entry.
- Every virtio function offers `VIRTIO_F_ACCESS_PLATFORM` when the machine has the IOMMU (`Transport::set_behind_iommu`). Without it a Linux driver hands the device bare guest-physical addresses, which the I/O page tables do not map, and every transfer is an IO_PAGE_FAULT (seen with virtio-net: TX timeouts). `pci.rs` tests drive all three device types against an IOMMU that blocks them and check that each fault is logged under its own requester id.
- IVRS (`acpi/mod.rs`): SELECT entries for 00:00.0, 00:00.2, every virtio devfn and the two special requester ids, then the special entries. Linux sizes the device table from the highest SELECT/range id only, so the special ids must also be SELECT entries or their table entries fall outside it.
- Under SVM/VMX nested guests the translation of the *host's* IOMMU is not involved; this is a model of an IOMMU the guest programs, nothing more.
- Not modeled: PPR, guest page tables (GT), ATS and IOTLB invalidation timeouts, the second event log, performance counters, per-device IRTE caching.

## Verified in a guest

Debian trixie's installer kernel (6.12.107, `AMD_IOMMU=y`) with `rdinit=/bin/sh` on `--iommu`: `AMD-Vi: Extended features (0x8c5, 0x0): PreF X2APIC IA GA`, `Interrupt remapping enabled`, `X2APIC enabled`, x2APIC cluster mode (logical destinations) through 128-bit IRTEs, `/proc/interrupts` shows `IR-IO-APIC` for the timer, ttyS0 and the SCI, `IR-PCI-MSIX` for virtio-net and `IOMMU-MSI` for the IOMMU's event, PPR and GA interrupts, and a `ping` through the user-mode gateway works with no IO_PAGE_FAULT. Alpine's virt kernel has no AMD IOMMU driver (only `virtio-iommu`), so it sees the IVRS table and ignores it. Not exercised in a guest: virtio-blk behind the IOMMU (Debian's installer initrd has no `virtio_blk`), only by the unit tests above.
