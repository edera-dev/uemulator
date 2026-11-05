use super::iommu::{Access, Iommu, PhysRead, PhysWrite};
#[cfg(freestanding)]
use crate::prelude::*;
use crate::ram::{RamSpan, is_ram_range};
use core::cell::RefCell;
use core::sync::atomic::{Ordering, fence};

/// Guest page size in bytes is 1 << PAGE_SHIFT; must match `Cpu::page_has_code` indexing.
const PAGE_SHIFT: u32 = 12;

/// Device-side view of guest RAM for DMA (virtqueues, request buffers).
///
/// Device code cannot take `&mut Cpu`, so this borrows RAM and the `page_has_code`
/// flags separately. DMA writes bypass the CPU store path (`Cpu::phys_write`), which is
/// what normally invalidates translated code, so every write that lands on a flagged
/// page is recorded in `code_pages_written`. The caller must drain `code_pages_written` into `Cpu::code_written`
/// once the borrow ends (see `tick_devices` and `phys_write` in cpu); skipping that
/// leaves stale decoded blocks or JIT code for memory the device just overwrote.
///
/// With `with_dma` the addresses are device addresses of one requester and go through the AMD
/// IOMMU first (see `dev/iommu`); `read_raw`/`write_raw` and `PhysRead`/`PhysWrite` always mean
/// physical addresses, which is what the IOMMU itself uses for its tables.
pub(crate) struct GuestMem<'a> {
    ram: RamSpan<'a>,
    /// `Cpu::page_has_code`: one byte per guest page, nonzero if a translated block covers it.
    code: &'a [u8],
    /// Page numbers written while flagged in `code`, each listed once.
    pub code_pages_written: Vec<usize>,
    /// Set only inside `with_dma`. A `RefCell` because `read` takes `&self` but a translation
    /// fills the IOTLB and queues fault events.
    dma: Option<DmaPort>,
}

struct DmaPort {
    iommu: RefCell<Iommu>,
    requester: u16,
}

/// A view of RAM for the IOMMU to read its tables through while `GuestMem` holds the IOMMU.
struct RamView<'r, 'a>(&'r RamSpan<'a>);

impl PhysRead for RamView<'_, '_> {
    fn read_phys(&self, phys_addr: u64, buf: &mut [u8]) -> bool {
        read_slice(self.0, phys_addr, buf)
    }
}

fn read_slice(ram: &RamSpan<'_>, phys_addr: u64, buf: &mut [u8]) -> bool {
    if !is_ram_range(ram.len() as u64, phys_addr, buf.len() as u64) {
        return false;
    }
    match usize::try_from(phys_addr)
        .ok()
        .and_then(|start| ram.get(start..start.checked_add(buf.len())?))
    {
        Some(src) => {
            buf.copy_from_slice(src);
            // A guest vCPU on another thread published what was just read (a virtqueue index)
            // with a release store; reads after this one must not be satisfied earlier.
            fence(Ordering::Acquire);
            true
        }
        None => false,
    }
}

/// How many bytes of `remaining` one translated mapping covers.
fn chunk_len(remaining: usize, bytes_left: u64) -> usize {
    remaining.min(usize::try_from(bytes_left).unwrap_or(usize::MAX))
}

impl<'a> GuestMem<'a> {
    pub fn new(ram: impl Into<RamSpan<'a>>, code: &'a [u8]) -> Self {
        GuestMem {
            ram: ram.into(),
            code,
            code_pages_written: Vec::new(),
            dma: None,
        }
    }

    /// Runs `f` with every access of this `GuestMem` translated by `iommu` as DMA from
    /// `requester` (bus/device/function). `iommu` is moved in for the duration and put back
    /// afterwards, so no lifetime ties it to the `GuestMem`.
    pub fn with_dma<R>(
        &mut self,
        iommu: &mut Iommu,
        requester: u16,
        f: impl FnOnce(&mut GuestMem<'a>) -> R,
    ) -> R {
        let previous = self.dma.replace(DmaPort {
            iommu: RefCell::new(core::mem::take(iommu)),
            requester,
        });
        let result = f(self);
        if let Some(port) = core::mem::replace(&mut self.dma, previous) {
            *iommu = port.iommu.into_inner();
        }
        result
    }

    /// True if `len` bytes at `phys_addr` are all RAM. Lets a device validate a guest-supplied
    /// buffer once up front instead of finding out per access. Under the IOMMU: every page of
    /// the range is mapped for this device (permissions are not checked, so a write buffer
    /// that is not readable still passes) and lands in RAM. A range that is not mapped logs the
    /// IOMMU's fault event, because the caller is about to refuse the device's request.
    pub fn contains(&self, phys_addr: u64, len: u64) -> bool {
        let Some(port) = &self.dma else {
            return self.contains_raw(phys_addr, len);
        };
        if len > self.ram.len() as u64 || phys_addr.checked_add(len).is_none() {
            return false;
        }
        let mut done = 0;
        while done < len {
            let probe = port.iommu.borrow_mut().probe(
                &RamView(&self.ram),
                port.requester,
                phys_addr + done,
            );
            let Some(translation) = probe else {
                // The device is about to be refused; let the IOMMU log why, as it would for
                // the access itself. Permissions never fail a probe, so this is a real fault.
                let _ = port.iommu.borrow_mut().translate(
                    &RamView(&self.ram),
                    port.requester,
                    phys_addr + done,
                    Access::Read,
                );
                return false;
            };
            let chunk = (len - done).min(translation.bytes_left);
            if !self.contains_raw(translation.phys, chunk) {
                return false;
            }
            done += chunk;
        }
        true
    }

    fn contains_raw(&self, phys_addr: u64, len: u64) -> bool {
        is_ram_range(self.ram.len() as u64, phys_addr, len)
    }

    /// Out-of-range reads fail instead of faulting: a bad guest descriptor must not panic the host.
    pub fn read(&self, phys_addr: u64, buf: &mut [u8]) -> bool {
        let Some(port) = &self.dma else {
            return self.read_raw(phys_addr, buf);
        };
        let mut done = 0;
        while done < buf.len() {
            let Some(address) = phys_addr.checked_add(done as u64) else {
                return false;
            };
            let translated = port.iommu.borrow_mut().translate(
                &RamView(&self.ram),
                port.requester,
                address,
                Access::Read,
            );
            let Ok(translation) = translated else {
                return false;
            };
            let chunk = chunk_len(buf.len() - done, translation.bytes_left);
            if !self.read_raw(translation.phys, &mut buf[done..done + chunk]) {
                return false;
            }
            done += chunk;
        }
        true
    }

    /// Returns false, writing nothing, if any byte of the range is outside RAM. Under the IOMMU
    /// a fault part way leaves the earlier pages written, as a real device's posted writes would
    /// be.
    pub fn write(&mut self, phys_addr: u64, data: &[u8]) -> bool {
        if self.dma.is_none() {
            return self.write_raw(phys_addr, data);
        }
        let mut done = 0;
        while done < data.len() {
            let Some(address) = phys_addr.checked_add(done as u64) else {
                return false;
            };
            let translated = match &self.dma {
                Some(port) => port.iommu.borrow_mut().translate(
                    &RamView(&self.ram),
                    port.requester,
                    address,
                    Access::Write,
                ),
                None => return false,
            };
            let Ok(translation) = translated else {
                return false;
            };
            let chunk = chunk_len(data.len() - done, translation.bytes_left);
            if !self.write_raw(translation.phys, &data[done..done + chunk]) {
                return false;
            }
            done += chunk;
        }
        true
    }

    /// `read` without the IOMMU: `phys_addr` is a physical address.
    pub fn read_raw(&self, phys_addr: u64, buf: &mut [u8]) -> bool {
        read_slice(&self.ram, phys_addr, buf)
    }

    /// `write` without the IOMMU: `phys_addr` is a physical address.
    pub fn write_raw(&mut self, phys_addr: u64, data: &[u8]) -> bool {
        let Ok(start) = usize::try_from(phys_addr) else {
            return false;
        };
        let Some(end) = start.checked_add(data.len()) else {
            return false;
        };
        if !is_ram_range(self.ram.len() as u64, phys_addr, data.len() as u64) {
            return false;
        }
        if data.is_empty() {
            return true;
        }
        self.ram[start..end].copy_from_slice(data);
        // Store first, then look at the flags: a vCPU that flags a page and decodes it after
        // this check sees the new bytes, and one that flagged it before is seen here (the other
        // half is the fence in `Cpu::flag_code_page_for_all`). The same fence orders this write
        // before the next one, which is how a device publishes a used-ring entry and then its
        // index to a guest on another thread.
        fence(Ordering::SeqCst);
        // A write can span pages; `end - 1` is the last byte written, not one past it.
        for page in (start >> PAGE_SHIFT)..=((end - 1) >> PAGE_SHIFT) {
            if self.code.get(page).is_some_and(|&flag| flag != 0)
                && !self.code_pages_written.contains(&page)
            {
                self.code_pages_written.push(page);
            }
        }
        true
    }

    /// The `read_u*` accessors return 0 for unmapped addresses.
    pub fn read_u16(&self, phys_addr: u64) -> u16 {
        let mut bytes = [0u8; 2];
        self.read(phys_addr, &mut bytes);
        u16::from_le_bytes(bytes)
    }

    pub fn read_u32(&self, phys_addr: u64) -> u32 {
        let mut bytes = [0u8; 4];
        self.read(phys_addr, &mut bytes);
        u32::from_le_bytes(bytes)
    }

    pub fn read_u64(&self, phys_addr: u64) -> u64 {
        let mut bytes = [0u8; 8];
        self.read(phys_addr, &mut bytes);
        u64::from_le_bytes(bytes)
    }

    pub fn write_u16(&mut self, phys_addr: u64, value: u16) {
        self.write(phys_addr, &value.to_le_bytes());
    }

    pub fn write_u32(&mut self, phys_addr: u64, value: u32) {
        self.write(phys_addr, &value.to_le_bytes());
    }
}

impl PhysRead for GuestMem<'_> {
    fn read_phys(&self, phys_addr: u64, buf: &mut [u8]) -> bool {
        self.read_raw(phys_addr, buf)
    }
}

impl PhysWrite for GuestMem<'_> {
    fn write_phys(&mut self, phys_addr: u64, data: &[u8]) -> bool {
        self.write_raw(phys_addr, data)
    }
}

#[cfg(test)]
pub(crate) fn test_mem(ram: &mut [u8]) -> GuestMem<'_> {
    GuestMem::new(ram, &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAM_BYTES: usize = 4 * (1 << PAGE_SHIFT);
    const PAGE_BYTES: u64 = 1 << PAGE_SHIFT;

    fn flags(pages: &[usize]) -> Vec<u8> {
        let mut map = vec![0u8; RAM_BYTES >> PAGE_SHIFT];
        for &page in pages {
            map[page] = 1;
        }
        map
    }

    #[test]
    fn write_then_read_roundtrips_bytes() {
        let mut ram = vec![0u8; RAM_BYTES];
        let mut mem = test_mem(&mut ram);
        assert!(mem.write(0x10, &[1, 2, 3, 4, 5]));
        let mut out = [0u8; 5];
        assert!(mem.read(0x10, &mut out));
        assert_eq!(out, [1, 2, 3, 4, 5]);
        assert_eq!(&ram[0x10..0x15], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn integer_accessors_are_little_endian() {
        let mut ram = vec![0u8; RAM_BYTES];
        ram[0x100..0x108].copy_from_slice(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
        let mut mem = test_mem(&mut ram);
        assert_eq!(mem.read_u16(0x100), 0x0201);
        assert_eq!(mem.read_u32(0x100), 0x0403_0201);
        assert_eq!(mem.read_u64(0x100), 0x0807_0605_0403_0201);
        mem.write_u16(0x200, 0xbeef);
        mem.write_u32(0x210, 0xdead_beef);
        assert_eq!(&ram[0x200..0x202], &[0xef, 0xbe]);
        assert_eq!(&ram[0x210..0x214], &[0xef, 0xbe, 0xad, 0xde]);
    }

    #[test]
    fn unaligned_accesses_work() {
        let mut ram = vec![0u8; RAM_BYTES];
        let mut mem = test_mem(&mut ram);
        mem.write_u32(0x3, 0x1122_3344);
        assert_eq!(mem.read_u32(0x3), 0x1122_3344);
        assert_eq!(mem.read_u16(0x4), 0x2233);
    }

    #[test]
    fn read_ending_exactly_at_ram_end_succeeds() {
        let mut ram = vec![0u8; RAM_BYTES];
        ram[RAM_BYTES - 1] = 0x7f;
        let mem = test_mem(&mut ram);
        let mut byte = [0u8; 1];
        assert!(mem.read(RAM_BYTES as u64 - 1, &mut byte));
        assert_eq!(byte, [0x7f]);
        assert_eq!(mem.read_u32(RAM_BYTES as u64 - 4), 0x7f00_0000);
    }

    #[test]
    fn read_crossing_ram_end_fails_without_touching_the_buffer() {
        let mut ram = vec![0u8; RAM_BYTES];
        let mem = test_mem(&mut ram);
        let mut buf = [0xaau8; 4];
        assert!(!mem.read(RAM_BYTES as u64 - 3, &mut buf));
        assert_eq!(buf, [0xaa; 4]);
    }

    #[test]
    fn read_starting_past_ram_fails() {
        let mut ram = vec![0u8; RAM_BYTES];
        let mem = test_mem(&mut ram);
        let mut buf = [0u8; 1];
        assert!(!mem.read(RAM_BYTES as u64, &mut buf));
        assert!(!mem.read(u64::MAX, &mut buf));
    }

    #[test]
    fn read_with_overflowing_range_fails_instead_of_panicking() {
        let mut ram = vec![0u8; RAM_BYTES];
        let mem = test_mem(&mut ram);
        let mut buf = [0u8; 16];
        assert!(!mem.read(u64::MAX - 7, &mut buf));
        assert!(!mem.read(usize::MAX as u64 - 7, &mut buf));
    }

    #[test]
    fn write_crossing_ram_end_writes_nothing() {
        let mut ram = vec![0u8; RAM_BYTES];
        let mut mem = test_mem(&mut ram);
        assert!(!mem.write(RAM_BYTES as u64 - 2, &[1, 2, 3, 4]));
        assert_eq!(&ram[RAM_BYTES - 4..], &[0, 0, 0, 0]);
    }

    #[test]
    fn write_past_ram_and_overflow_fail() {
        let mut ram = vec![0u8; RAM_BYTES];
        let mut mem = test_mem(&mut ram);
        assert!(!mem.write(RAM_BYTES as u64, &[1]));
        assert!(!mem.write(u64::MAX, &[1, 2]));
        assert!(!mem.write(u64::MAX - 1, &[1, 2, 3]));
    }

    #[test]
    fn empty_write_inside_ram_succeeds_and_outside_fails() {
        let mut ram = vec![0u8; RAM_BYTES];
        let mut mem = test_mem(&mut ram);
        assert!(mem.write(0, &[]));
        assert!(mem.write(RAM_BYTES as u64, &[]));
        assert!(!mem.write(RAM_BYTES as u64 + 1, &[]));
        assert!(mem.code_pages_written.is_empty());
    }

    #[test]
    fn out_of_range_integer_reads_return_zero_and_writes_are_dropped() {
        let mut ram = vec![0xffu8; RAM_BYTES];
        let mut mem = test_mem(&mut ram);
        assert_eq!(mem.read_u16(RAM_BYTES as u64), 0);
        assert_eq!(mem.read_u32(RAM_BYTES as u64 - 2), 0);
        assert_eq!(mem.read_u64(u64::MAX), 0);
        mem.write_u32(RAM_BYTES as u64 - 2, 0);
        assert_eq!(&ram[RAM_BYTES - 4..], &[0xff; 4]);
    }

    #[test]
    fn contains_is_inclusive_of_the_last_byte_and_overflow_safe() {
        let mut ram = vec![0u8; RAM_BYTES];
        let mem = test_mem(&mut ram);
        assert!(mem.contains(0, RAM_BYTES as u64));
        assert!(mem.contains(RAM_BYTES as u64, 0));
        assert!(mem.contains(RAM_BYTES as u64 - 1, 1));
        assert!(!mem.contains(RAM_BYTES as u64 - 1, 2));
        assert!(!mem.contains(RAM_BYTES as u64 + 1, 0));
        assert!(!mem.contains(u64::MAX, 2));
        assert!(!mem.contains(1, u64::MAX));
    }

    #[test]
    fn write_to_unflagged_page_records_nothing() {
        let mut ram = vec![0u8; RAM_BYTES];
        let map = flags(&[2]);
        let mut mem = GuestMem::new(&mut ram, &map);
        mem.write(0x10, &[1]);
        mem.write(PAGE_BYTES, &[1]);
        assert!(mem.code_pages_written.is_empty());
    }

    #[test]
    fn write_to_flagged_page_records_that_page() {
        let mut ram = vec![0u8; RAM_BYTES];
        let map = flags(&[1]);
        let mut mem = GuestMem::new(&mut ram, &map);
        mem.write(PAGE_BYTES + 8, &[1, 2]);
        assert_eq!(mem.code_pages_written, vec![1]);
    }

    #[test]
    fn repeated_writes_to_a_flagged_page_are_listed_once() {
        let mut ram = vec![0u8; RAM_BYTES];
        let map = flags(&[3]);
        let mut mem = GuestMem::new(&mut ram, &map);
        for offset in 0..4 {
            mem.write(3 * PAGE_BYTES + offset, &[9]);
        }
        assert_eq!(mem.code_pages_written, vec![3]);
    }

    #[test]
    fn write_spanning_two_flagged_pages_records_both() {
        let mut ram = vec![0u8; RAM_BYTES];
        let map = flags(&[0, 1]);
        let mut mem = GuestMem::new(&mut ram, &map);
        mem.write(PAGE_BYTES - 2, &[1, 2, 3, 4]);
        assert_eq!(mem.code_pages_written, vec![0, 1]);
    }

    #[test]
    fn write_ending_exactly_at_page_boundary_does_not_touch_the_next_page() {
        let mut ram = vec![0u8; RAM_BYTES];
        let map = flags(&[1]);
        let mut mem = GuestMem::new(&mut ram, &map);
        mem.write(PAGE_BYTES - 4, &[1, 2, 3, 4]);
        assert!(mem.code_pages_written.is_empty());
        mem.write(PAGE_BYTES, &[1]);
        assert_eq!(mem.code_pages_written, vec![1]);
    }

    #[test]
    fn integer_writes_track_code_pages_too() {
        let mut ram = vec![0u8; RAM_BYTES];
        let map = flags(&[2]);
        let mut mem = GuestMem::new(&mut ram, &map);
        mem.write_u16(2 * PAGE_BYTES, 1);
        mem.write_u32(2 * PAGE_BYTES + 4, 1);
        assert_eq!(mem.code_pages_written, vec![2]);
    }

    #[test]
    fn failed_write_does_not_record_a_code_page() {
        let mut ram = vec![0u8; RAM_BYTES];
        let map = flags(&[3]);
        let mut mem = GuestMem::new(&mut ram, &map);
        assert!(!mem.write(4 * PAGE_BYTES - 2, &[1, 2, 3, 4]));
        assert!(mem.code_pages_written.is_empty());
    }

    #[test]
    fn reads_never_record_code_pages() {
        let mut ram = vec![0u8; RAM_BYTES];
        let map = flags(&[0, 1, 2, 3]);
        let mem = GuestMem::new(&mut ram, &map);
        let mut buf = [0u8; 64];
        mem.read(0, &mut buf);
        assert!(mem.code_pages_written.is_empty());
    }

    #[test]
    fn page_beyond_the_flag_map_is_treated_as_not_code() {
        let mut ram = vec![0u8; RAM_BYTES];
        let map = flags(&[0]);
        let short_map = &map[..2];
        let mut mem = GuestMem::new(&mut ram, short_map);
        mem.write(3 * PAGE_BYTES, &[1]);
        assert!(mem.code_pages_written.is_empty());
    }
}
