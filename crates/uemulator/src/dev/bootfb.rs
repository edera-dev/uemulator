//! The boot framebuffer: a linear framebuffer in guest RAM that the UEFI firmware publishes as
//! its Graphics Output Protocol, and that the screen shows until the guest's virtio-gpu driver
//! sets a scanout. It is QEMU's `ramfb` idea next to a virtio-gpu (`virtio-ramfb`): plain RAM,
//! so the firmware and an EFI-handed `efifb`/`simpledrm` draw at memory speed, where a BAR
//! would trap every store into the device model. The cost is that nothing tells the device what
//! changed, so `tick` compares the RAM with the screen at most every `SCAN_INTERVAL_NS`.
//!
//! The region is carved out of the top of the RAM below the PCI hole and reported as reserved
//! in the firmware's memory map (`boot/uefi.rs`), so neither the firmware's allocator nor the OS
//! puts anything else there.

use super::guestmem::GuestMem;
use crate::display::{BYTES_PER_PIXEL, DisplayHandle, Rect};
#[cfg(freestanding)]
use crate::prelude::*;

/// How often the RAM is compared with the screen, in guest nanoseconds: 30 times a second, which
/// is smooth for text and menus. A scan reads the whole framebuffer (4 MB at 1280x800), so much
/// more often would show up in the host's CPU use.
const SCAN_INTERVAL_NS: u64 = 1_000_000_000 / 30;

/// Where the firmware's framebuffer is and how it is laid out. Pixels are always four bytes,
/// blue, green, red, unused (GOP's `PixelBlueGreenRedReserved8BitPerColor`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FramebufferRegion {
    pub base: u64,
    pub width: u32,
    pub height: u32,
}

impl FramebufferRegion {
    pub fn stride(&self) -> usize {
        self.width as usize * BYTES_PER_PIXEL
    }

    /// Bytes the pixels occupy.
    pub fn pixel_bytes(&self) -> u64 {
        self.stride() as u64 * self.height as u64
    }
}

pub(crate) struct BootFramebuffer {
    region: FramebufferRegion,
    display: DisplayHandle,
    /// Guest time of the last scan; `None` until the first.
    last_scan: Option<u64>,
    row: Vec<u8>,
}

impl BootFramebuffer {
    pub fn new(region: FramebufferRegion, display: DisplayHandle) -> Self {
        BootFramebuffer {
            region,
            display,
            last_scan: None,
            row: vec![0; region.stride()],
        }
    }

    /// Copies the rows that differ from the screen into it, unless the virtio-gpu driver owns
    /// the screen or the last scan was less than `SCAN_INTERVAL_NS` ago. Reads physical RAM
    /// directly: the region is a CPU address, never translated by an IOMMU.
    pub fn tick(&mut self, now: u64, mem: &GuestMem) {
        if self.display.virtio_owns_screen() {
            // Scan in full again when the screen comes back: it holds the driver's image now.
            self.last_scan = None;
            return;
        }
        if self
            .last_scan
            .is_some_and(|last| now.saturating_sub(last) < SCAN_INTERVAL_NS)
        {
            return;
        }
        self.last_scan = Some(now);
        let region = self.region;
        let row = &mut self.row;
        self.display
            .publish(region.width, region.height, |screen, stride| {
                let mut changed: Option<(u32, u32)> = None;
                for line in 0..region.height {
                    let address = region.base + line as u64 * region.stride() as u64;
                    if !mem.read_raw(address, row) {
                        break;
                    }
                    let target = line as usize * stride;
                    let shown = &mut screen[target..target + row.len()];
                    if shown != row.as_slice() {
                        shown.copy_from_slice(row);
                        changed = Some(match changed {
                            Some((first, _)) => (first, line),
                            None => (line, line),
                        });
                    }
                }
                changed.map(|(first, last)| Rect {
                    x: 0,
                    y: first,
                    width: region.width,
                    height: last - first + 1,
                })
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::guestmem::test_mem;

    const BASE: u64 = 0x1000;
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 4;
    const RAM_BYTES: usize = 0x2000;

    fn region() -> FramebufferRegion {
        FramebufferRegion {
            base: BASE,
            width: WIDTH,
            height: HEIGHT,
        }
    }

    fn pixel_address(x: u32, y: u32) -> usize {
        BASE as usize + (y * WIDTH + x) as usize * BYTES_PER_PIXEL
    }

    #[test]
    fn the_first_scan_shows_the_framebuffer_at_its_size() {
        let display = DisplayHandle::with_size(640, 480);
        let mut ram = vec![0u8; RAM_BYTES];
        ram[pixel_address(1, 2)] = 0xaa;
        let mut framebuffer = BootFramebuffer::new(region(), display.clone());
        framebuffer.tick(0, &test_mem(&mut ram));
        assert_eq!(display.size(), (WIDTH, HEIGHT));
        let frame = display.frame();
        assert_eq!(
            frame.pixels[(2 * WIDTH + 1) as usize * BYTES_PER_PIXEL],
            0xaa
        );
        assert_eq!(display.generation(), 1);
    }

    #[test]
    fn only_changed_rows_are_reported_and_an_unchanged_framebuffer_reports_nothing() {
        let display = DisplayHandle::with_size(WIDTH, HEIGHT);
        let mut ram = vec![0u8; RAM_BYTES];
        let mut framebuffer = BootFramebuffer::new(region(), display.clone());
        framebuffer.tick(0, &test_mem(&mut ram));
        assert_eq!(display.generation(), 0, "black on black is no change");
        ram[pixel_address(3, 1)] = 1;
        ram[pixel_address(0, 2)] = 2;
        framebuffer.tick(SCAN_INTERVAL_NS, &test_mem(&mut ram));
        let mut pixels = Vec::new();
        let update = display.update_since(0, &mut pixels).unwrap();
        assert_eq!(
            update.rect,
            Rect {
                x: 0,
                y: 1,
                width: WIDTH,
                height: 2
            }
        );
        framebuffer.tick(2 * SCAN_INTERVAL_NS, &test_mem(&mut ram));
        assert_eq!(display.generation(), update.generation);
    }

    #[test]
    fn scans_are_spaced_by_the_interval() {
        let display = DisplayHandle::with_size(WIDTH, HEIGHT);
        let mut ram = vec![0u8; RAM_BYTES];
        let mut framebuffer = BootFramebuffer::new(region(), display.clone());
        framebuffer.tick(1000, &test_mem(&mut ram));
        ram[pixel_address(0, 0)] = 9;
        framebuffer.tick(1000 + SCAN_INTERVAL_NS - 1, &test_mem(&mut ram));
        assert_eq!(display.generation(), 0, "too soon");
        framebuffer.tick(1000 + SCAN_INTERVAL_NS, &test_mem(&mut ram));
        assert_eq!(display.generation(), 1);
    }

    #[test]
    fn nothing_is_scanned_while_the_virtio_driver_owns_the_screen() {
        let display = DisplayHandle::with_size(WIDTH, HEIGHT);
        let mut ram = vec![0u8; RAM_BYTES];
        ram[pixel_address(0, 0)] = 9;
        display.set_virtio_owns_screen(true);
        let mut framebuffer = BootFramebuffer::new(region(), display.clone());
        framebuffer.tick(0, &test_mem(&mut ram));
        assert_eq!(display.generation(), 0);
        display.set_virtio_owns_screen(false);
        framebuffer.tick(1, &test_mem(&mut ram));
        assert_eq!(
            display.generation(),
            1,
            "rescanned at once when handed back"
        );
    }

    #[test]
    fn a_region_outside_ram_shows_nothing_and_does_not_panic() {
        let display = DisplayHandle::with_size(WIDTH, HEIGHT);
        let mut ram = vec![0u8; RAM_BYTES];
        let outside = FramebufferRegion {
            base: RAM_BYTES as u64 - 16,
            ..region()
        };
        let mut framebuffer = BootFramebuffer::new(outside, display.clone());
        framebuffer.tick(0, &test_mem(&mut ram));
        assert_eq!(display.generation(), 0);
    }
}
