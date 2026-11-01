//! A polling driver for modern virtio-blk over PCI, enough for the boot path: one queue, one
//! request in flight, no interrupts. The device model is `crates/uemulator/src/dev/virtio.rs`.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ptr::{read_volatile, write_volatile};

use crate::memory;
use crate::pci::{self, Function};
use crate::uefi::memory_type::BOOT_SERVICES_DATA;
use crate::virtio_pci::{DEVICE_ID_BASE, Transport, VENDOR_VIRTIO, write16};

/// Modern block device: device type 2.
const DEVICE_ID_BLOCK: u16 = DEVICE_ID_BASE + 2;
pub const SECTOR_BYTES: usize = 512;
/// Largest transfer in one request; the device allows 16 MiB, this keeps buffers modest.
const MAX_REQUEST_BYTES: usize = 1 << 20;
const FEATURE_RO: u32 = 1 << 5;

const QUEUE_SIZE: u16 = 16;
const DESC_BYTES: usize = 16;
const DESC_F_NEXT: u16 = 1;
const DESC_F_WRITE: u16 = 2;
const REQUEST_IN: u32 = 0;
const REQUEST_OUT: u32 = 1;
const REQUEST_FLUSH: u32 = 4;
const HEADER_BYTES: usize = 16;
const STATUS_OK: u8 = 0;
/// The device answers during the notify write in this emulator; this bound is for a device that
/// never does.
const POLL_LIMIT: u32 = 50_000_000;
const DEVICE_CONFIG_CAPACITY: usize = 0;

/// One page for the request header and status byte, then the rings.
const RING_PAGES: u64 = 3;
const SCRATCH_OFFSET: usize = 0;
const AVAIL_OFFSET: usize = DESC_BYTES * QUEUE_SIZE as usize;
const USED_OFFSET: usize = 4096;
const STATUS_BYTE_OFFSET: usize = 256;
const SCRATCH_PAGE: usize = 8192;

pub struct VirtioBlk {
    pub function: Function,
    transport: Transport,
    notify: *mut u8,
    ring: *mut u8,
    pub sector_count: u64,
    pub read_only: bool,
    last_used: u16,
    available: u16,
}

/// The virtio block devices on the bus, initialised and ready.
// Boxed: the disks are leaked and BlockIO devices keep their addresses.
#[allow(clippy::vec_box)]
pub fn probe_all() -> Vec<Box<VirtioBlk>> {
    pci::scan()
        .into_iter()
        .filter(|function| {
            function.vendor == VENDOR_VIRTIO && function.device_id == DEVICE_ID_BLOCK
        })
        .filter_map(VirtioBlk::new)
        .map(Box::new)
        .collect()
}

impl VirtioBlk {
    fn new(function: Function) -> Option<Self> {
        let transport = Transport::new(&function)?;
        let ring = memory::allocate_zeroed_pages(BOOT_SERVICES_DATA, RING_PAGES)?;
        let features = transport.negotiate(0)?;
        let notify = transport.setup_queue(0, QUEUE_SIZE, ring, AVAIL_OFFSET, USED_OFFSET)?;
        transport.driver_ok();
        let sector_count = unsafe {
            read_volatile(
                transport
                    .device_config
                    .add(DEVICE_CONFIG_CAPACITY)
                    .cast::<u64>(),
            )
        };
        Some(Self {
            function,
            transport,
            notify,
            ring,
            sector_count,
            read_only: features & FEATURE_RO != 0,
            last_used: 0,
            available: 0,
        })
    }

    /// Stops the device, so the OS finds it as it would after power-on.
    pub fn reset(&mut self) {
        self.transport.reset();
    }

    /// One request: header, optional data buffer, status byte. `device_writes` says whether the
    /// device fills the buffer (a read) or only reads it (a write).
    fn submit(
        &mut self,
        kind: u32,
        sector: u64,
        data: *mut u8,
        length: usize,
        device_writes: bool,
    ) -> bool {
        unsafe {
            let scratch = self.ring.add(SCRATCH_PAGE + SCRATCH_OFFSET);
            scratch.cast::<u32>().write(kind);
            scratch.add(4).cast::<u32>().write(0);
            scratch.add(8).cast::<u64>().write(sector);
            let status = scratch.add(STATUS_BYTE_OFFSET);
            status.write(0xff);
            let descriptor = |index: usize, address: u64, length: u32, flags: u16, next: u16| {
                let at = self.ring.add(index * DESC_BYTES);
                at.cast::<u64>().write(address);
                at.add(8).cast::<u32>().write(length);
                at.add(12).cast::<u16>().write(flags);
                at.add(14).cast::<u16>().write(next);
            };
            let header_flags = DESC_F_NEXT;
            descriptor(
                0,
                scratch as u64,
                HEADER_BYTES as u32,
                header_flags,
                if length > 0 { 1 } else { 2 },
            );
            if length > 0 {
                let flags = DESC_F_NEXT | if device_writes { DESC_F_WRITE } else { 0 };
                descriptor(1, data as u64, length as u32, flags, 2);
            }
            descriptor(2, status as u64, 1, DESC_F_WRITE, 0);
            let avail = self.ring.add(AVAIL_OFFSET);
            let slot = (self.available % QUEUE_SIZE) as usize;
            avail.add(4 + 2 * slot).cast::<u16>().write(0);
            self.available = self.available.wrapping_add(1);
            write_volatile(avail.add(2).cast::<u16>(), self.available);
            write16(self.notify, 0, 0);
            let used = self.ring.add(USED_OFFSET);
            for _ in 0..POLL_LIMIT {
                if read_volatile(used.add(2).cast::<u16>()) != self.last_used {
                    self.last_used = self.last_used.wrapping_add(1);
                    return read_volatile(status) == STATUS_OK;
                }
                core::hint::spin_loop();
            }
            false
        }
    }

    pub fn read(&mut self, first_sector: u64, buffer: &mut [u8]) -> bool {
        if !buffer.len().is_multiple_of(SECTOR_BYTES)
            || first_sector + (buffer.len() / SECTOR_BYTES) as u64 > self.sector_count
        {
            return false;
        }
        let mut sector = first_sector;
        for chunk in buffer.chunks_mut(MAX_REQUEST_BYTES) {
            if !self.submit(REQUEST_IN, sector, chunk.as_mut_ptr(), chunk.len(), true) {
                return false;
            }
            sector += (chunk.len() / SECTOR_BYTES) as u64;
        }
        true
    }

    pub fn write(&mut self, first_sector: u64, buffer: &[u8]) -> bool {
        if self.read_only
            || !buffer.len().is_multiple_of(SECTOR_BYTES)
            || first_sector + (buffer.len() / SECTOR_BYTES) as u64 > self.sector_count
        {
            return false;
        }
        let mut sector = first_sector;
        for chunk in buffer.chunks(MAX_REQUEST_BYTES) {
            if !self.submit(
                REQUEST_OUT,
                sector,
                chunk.as_ptr().cast_mut(),
                chunk.len(),
                false,
            ) {
                return false;
            }
            sector += (chunk.len() / SECTOR_BYTES) as u64;
        }
        true
    }

    pub fn flush(&mut self) -> bool {
        self.submit(REQUEST_FLUSH, 0, core::ptr::null_mut(), 0, false)
    }
}
