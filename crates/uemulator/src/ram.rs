//! Guest RAM and the per-page "translated code" flags, as handles that every vCPU of a machine
//! clones. One allocation backs all of them, so with host threads the vCPUs see each other's
//! stores, and with one thread nothing changes: the ranged views below behave like a `Vec<u8>`.
//!
//! The views hand out `&[u8]` and `&mut [u8]` over memory other vCPU threads may be writing. That
//! is a data race in Rust's model, and it is what the guest asked for: a racing guest sees
//! whichever store wins, as on hardware. Code that needs more than "some value" (page-table
//! entries, atomic read-modify-write, anything a host loop polls) goes through the raw accessors
//! in `cpu/atomic.rs` instead, and never keeps a slice across a point where another thread could
//! change what it depends on.
//!
//! There is deliberately no view of all of RAM as one slice. On wasm32 the RAM can be larger than
//! `isize::MAX` (3 GiB on the web page), and Rust allows no slice, `pointer::add` or allocation
//! that large: the compiler marks slice indexing as an in-bounds offset that a signed 32-bit
//! index past 2 GiB would break. So every view is a range checked here, its address formed with
//! `wrapping_add`, and code that walks RAM asks for the piece it needs.

#[cfg(freestanding)]
use crate::prelude::*;
use crate::sync::Arc;
use alloc::alloc::{Layout, alloc_zeroed};
use core::marker::PhantomData;
use core::ops::{Index, IndexMut, Range, RangeFrom};
use core::sync::atomic::{AtomicU8, Ordering};

/// First guest physical address of the hole below 4 GiB that holds the PCI BARs, the ECAM, the
/// I/O APIC and the local APIC. RAM that does not fit below it continues at `PCI_HOLE_END`.
pub(crate) const PCI_HOLE_START: u64 = 0xc000_0000;
pub(crate) const PCI_HOLE_END: u64 = 0x1_0000_0000;

/// Length of the flat buffer that backs `memory_bytes` of guest RAM. Guest physical address `a`
/// is byte `a` of the buffer, so with more RAM than fits below the hole the buffer spans the
/// hole as well: its `PCI_HOLE_END - PCI_HOLE_START` bytes are never touched, and the host
/// commits memory lazily, so they cost address space and nothing else. Every "is this RAM"
/// question must go through `is_ram_range` rather than compare against the buffer length, which
/// the hole would answer wrongly.
pub(crate) fn flat_bytes(memory_bytes: u64) -> u64 {
    if memory_bytes <= PCI_HOLE_START {
        memory_bytes
    } else {
        PCI_HOLE_END + (memory_bytes - PCI_HOLE_START)
    }
}

/// End of the RAM below the hole, for a buffer of `flat_len` bytes.
pub(crate) fn low_ram_end(flat_len: u64) -> u64 {
    flat_len.min(PCI_HOLE_START)
}

/// End of the RAM above 4 GiB, or 0 when there is none.
pub(crate) fn high_ram_end(flat_len: u64) -> u64 {
    if flat_len > PCI_HOLE_END { flat_len } else { 0 }
}

/// Whether all of `start..start + len` is RAM, for a buffer of `flat_len` bytes.
pub(crate) fn is_ram_range(flat_len: u64, start: u64, len: u64) -> bool {
    let Some(end) = start.checked_add(len) else {
        return false;
    };
    end <= flat_len && (end <= PCI_HOLE_START || start >= PCI_HOLE_END)
}

/// `len` zeroed bytes, or `None` when the host refuses. Large sizes come from the allocator's
/// zeroed path (`calloc`, which maps fresh pages lazily), where `vec![0; len]` would be the same
/// but aborts the process on failure and a guest's memory size is a user setting.
pub(crate) fn zeroed_bytes(len: usize) -> Option<Vec<u8>> {
    if len == 0 {
        return Some(Vec::new());
    }
    let layout = Layout::array::<u8>(len).ok()?;
    let pointer = unsafe { alloc_zeroed(layout) };
    if pointer.is_null() {
        return None;
    }
    // Allocated with the global allocator at exactly this layout, which is what `Vec<u8>` frees.
    Some(unsafe { Vec::from_raw_parts(pointer, len, len) })
}

/// What keeps a `GuestRam`'s memory alive.
enum Backing {
    /// From the global allocator. Never touched except to free it on drop; a `Vec` does not move
    /// its heap buffer, so the block's pointer stays valid.
    Allocated(#[allow(dead_code)] Vec<u8>),
    /// Pages added to the module's memory with `memory.grow`, for RAM the allocator cannot hand
    /// out (more than `isize::MAX` bytes). WebAssembly memory never shrinks, so these are never
    /// freed: the web page runs one machine per instance, and `GuestRam::zeroed_again` reuses
    /// them across a guest reboot instead of growing a second copy that would not fit.
    #[cfg(target_arch = "wasm32")]
    Grown,
}

/// The memory behind a `GuestRam`.
struct RamBlock {
    pointer: *mut u8,
    length: usize,
    // Read only on wasm32, the one host with `Backing::Grown`; elsewhere it only frees the RAM.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    backing: Backing,
}

// The block is plain memory; sharing it between threads is the point.
unsafe impl Send for RamBlock {}
unsafe impl Sync for RamBlock {}

/// Guest physical RAM starting at address 0. Cloning shares the memory.
#[derive(Clone)]
pub(crate) struct GuestRam {
    block: Arc<RamBlock>,
}

impl GuestRam {
    pub fn new(mut bytes: Vec<u8>) -> Self {
        let pointer = bytes.as_mut_ptr();
        let length = bytes.len();
        GuestRam {
            block: Arc::new(RamBlock {
                pointer,
                length,
                backing: Backing::Allocated(bytes),
            }),
        }
    }

    /// `length` zeroed bytes, or `None` when the host refuses.
    pub fn zeroed(length: usize) -> Option<Self> {
        if length <= isize::MAX as usize {
            return zeroed_bytes(length).map(GuestRam::new);
        }
        #[cfg(target_arch = "wasm32")]
        {
            let pointer = grow_zeroed(length)?;
            Some(GuestRam {
                block: Arc::new(RamBlock {
                    pointer,
                    length,
                    backing: Backing::Grown,
                }),
            })
        }
        #[cfg(not(target_arch = "wasm32"))]
        None
    }

    /// The same amount of zeroed RAM, for a reboot. Fresh pages from the allocator rather than a
    /// fill of the old ones, which would commit every page the host had lazily left untouched;
    /// grown pages cannot be given back, so those are zeroed in place, which needs this to be the
    /// last handle.
    pub fn zeroed_again(self) -> Option<Self> {
        let length = self.len();
        #[cfg(target_arch = "wasm32")]
        if let Backing::Grown = self.block.backing {
            let block = Arc::into_inner(self.block)?;
            let mut ram = GuestRam {
                block: Arc::new(block),
            };
            for range in views_covering(length) {
                ram[range].fill(0);
            }
            return Some(ram);
        }
        drop(self);
        GuestRam::zeroed(length)
    }

    /// A copy that shares nothing with this RAM, for `verify`'s interpreter twin.
    pub fn duplicate(&self) -> Option<Self> {
        let mut copy = GuestRam::zeroed(self.len())?;
        for range in views_covering(self.len()) {
            copy[range.clone()].copy_from_slice(&self[range]);
        }
        Some(copy)
    }

    /// Host address of byte 0.
    pub fn base(&self) -> *mut u8 {
        self.block.pointer
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.block.pointer
    }

    /// Whether all of `start..start + len` is RAM (not past the end, not in the PCI hole).
    pub fn contains(&self, start: u64, len: u64) -> bool {
        is_ram_range(self.block.length as u64, start, len)
    }

    /// `len` bytes at `start`, or `None` where `contains` is false.
    pub fn slice(&self, start: u64, len: usize) -> Option<&[u8]> {
        if !self.contains(start, len as u64) {
            return None;
        }
        self.get(start as usize..start as usize + len)
    }

    fn extent(&self) -> (*mut u8, usize) {
        (self.block.pointer, self.block.length)
    }
}

/// The longest view `GuestRam` hands out: the most a Rust slice may cover.
const MAX_VIEW_BYTES: usize = isize::MAX as usize;

/// `0..length` as ranges no longer than a view may be.
fn views_covering(length: usize) -> impl Iterator<Item = Range<usize>> {
    (0..length)
        .step_by(MAX_VIEW_BYTES)
        .map(move |start| start..length.min(start + MAX_VIEW_BYTES))
}

/// The range `ram[start..]` means: to the end, but no longer than a slice may be. Only RAM past
/// `isize::MAX` (wasm32) is ever cut short, and the callers take a short prefix of it
/// (`ram[start..][..len]`), so a cut view still holds what they index.
fn capped(range: RangeFrom<usize>, length: usize) -> Range<usize> {
    range.start..length.min(range.start.saturating_add(MAX_VIEW_BYTES))
}

/// `length` zeroed bytes grown onto the module's memory, or `None` when it cannot grow that far.
/// New WebAssembly pages are zero, so nothing is written.
#[cfg(target_arch = "wasm32")]
fn grow_zeroed(length: usize) -> Option<*mut u8> {
    const WASM_PAGE_BYTES: usize = 64 << 10;
    const MAIN_MEMORY: u32 = 0;
    const GROW_FAILED: usize = usize::MAX;
    let first_page =
        core::arch::wasm32::memory_grow::<MAIN_MEMORY>(length.div_ceil(WASM_PAGE_BYTES));
    if first_page == GROW_FAILED {
        return None;
    }
    // Memory from `memory.grow` belongs to no Rust allocation; it is addressed like memory a host
    // mapped, by its address.
    Some(core::ptr::with_exposed_provenance_mut(
        first_page * WASM_PAGE_BYTES,
    ))
}

impl Default for GuestRam {
    fn default() -> Self {
        GuestRam::new(Vec::new())
    }
}

impl From<Vec<u8>> for GuestRam {
    fn from(bytes: Vec<u8>) -> Self {
        GuestRam::new(bytes)
    }
}

/// Guest RAM borrowed for writing: what the boot loaders and device DMA (`GuestMem`) take. Made
/// from a `GuestRam`, or from a buffer of the host's own (`load_linux`, `Platform::tick`, the
/// device tests), which can never be larger than a slice.
pub(crate) struct RamSpan<'a> {
    pointer: *mut u8,
    length: usize,
    _borrow: PhantomData<&'a mut [u8]>,
}

impl RamSpan<'_> {
    /// A shorter-lived span over the same bytes, to hand to a callee and keep this one.
    pub fn reborrow(&mut self) -> RamSpan<'_> {
        RamSpan {
            pointer: self.pointer,
            length: self.length,
            _borrow: PhantomData,
        }
    }

    fn extent(&self) -> (*mut u8, usize) {
        (self.pointer, self.length)
    }
}

impl<'a> From<&'a mut [u8]> for RamSpan<'a> {
    fn from(bytes: &'a mut [u8]) -> Self {
        RamSpan {
            pointer: bytes.as_mut_ptr(),
            length: bytes.len(),
            _borrow: PhantomData,
        }
    }
}

impl<'a> From<&'a mut Vec<u8>> for RamSpan<'a> {
    fn from(bytes: &'a mut Vec<u8>) -> Self {
        RamSpan::from(bytes.as_mut_slice())
    }
}

impl<'a> From<&'a mut GuestRam> for RamSpan<'a> {
    fn from(ram: &'a mut GuestRam) -> Self {
        let (pointer, length) = ram.extent();
        RamSpan {
            pointer,
            length,
            _borrow: PhantomData,
        }
    }
}

/// Where `range` of the `length` bytes at `pointer` starts, or `None` when it is reversed, runs
/// past the end, or is longer than a Rust slice may be. The PCI hole is not checked: it is part
/// of the buffer, just never touched.
fn view_start(pointer: *mut u8, length: usize, range: &Range<usize>) -> Option<*mut u8> {
    let fits = range.start <= range.end
        && range.end <= length
        && range.end - range.start <= MAX_VIEW_BYTES;
    // `wrapping_add`, not `add`: the offset may be past `isize::MAX` (see the module comment).
    fits.then(|| pointer.wrapping_add(range.start))
}

#[track_caller]
fn out_of_range(range: &Range<usize>, length: usize) -> ! {
    panic!(
        "guest RAM range {range:?} is reversed, past its {length} bytes or longer than {MAX_VIEW_BYTES}"
    )
}

/// The ranged views of `GuestRam` and `RamSpan`, which both reduce to `extent`. Indexing panics
/// like a slice's would; `ram[start..]` is `capped`.
macro_rules! ram_views {
    ($ram:ty) => {
        impl $ram {
            pub fn len(&self) -> usize {
                self.extent().1
            }

            pub fn get(&self, range: Range<usize>) -> Option<&[u8]> {
                let (pointer, length) = self.extent();
                let start = view_start(pointer, length, &range)?;
                Some(unsafe { core::slice::from_raw_parts(start, range.end - range.start) })
            }

            pub fn get_mut(&mut self, range: Range<usize>) -> Option<&mut [u8]> {
                let (pointer, length) = self.extent();
                let start = view_start(pointer, length, &range)?;
                Some(unsafe { core::slice::from_raw_parts_mut(start, range.end - range.start) })
            }
        }

        impl Index<Range<usize>> for $ram {
            type Output = [u8];
            #[track_caller]
            fn index(&self, range: Range<usize>) -> &[u8] {
                match self.get(range.clone()) {
                    Some(bytes) => bytes,
                    None => out_of_range(&range, self.len()),
                }
            }
        }

        impl IndexMut<Range<usize>> for $ram {
            #[track_caller]
            fn index_mut(&mut self, range: Range<usize>) -> &mut [u8] {
                let length = self.len();
                match self.get_mut(range.clone()) {
                    Some(bytes) => bytes,
                    None => out_of_range(&range, length),
                }
            }
        }

        impl Index<RangeFrom<usize>> for $ram {
            type Output = [u8];
            #[track_caller]
            fn index(&self, range: RangeFrom<usize>) -> &[u8] {
                &self[capped(range, self.len())]
            }
        }

        impl IndexMut<RangeFrom<usize>> for $ram {
            #[track_caller]
            fn index_mut(&mut self, range: RangeFrom<usize>) -> &mut [u8] {
                let length = self.len();
                &mut self[capped(range, length)]
            }
        }

        impl Index<usize> for $ram {
            type Output = u8;
            #[track_caller]
            fn index(&self, offset: usize) -> &u8 {
                &self[offset..offset.wrapping_add(1)][0]
            }
        }

        impl IndexMut<usize> for $ram {
            #[track_caller]
            fn index_mut(&mut self, offset: usize) -> &mut u8 {
                &mut self[offset..offset.wrapping_add(1)][0]
            }
        }
    };
}

ram_views!(GuestRam);
ram_views!(RamSpan<'_>);

/// One byte per guest physical page: nonzero while some vCPU's decode cache or JIT holds code
/// built from that page. Cloning shares the flags.
///
/// A store to a flagged page must reach `Cpu::code_written`; the soft TLB withholds its write tag
/// for such pages so the store takes the slow path. With several threads a flag is set by one
/// vCPU and read by the others, so the accessors are atomic (relaxed: callers that need ordering
/// against guest data add the fences themselves, see `Cpu::mark_code`).
#[derive(Clone)]
pub(crate) struct CodePages {
    flags: Arc<Vec<AtomicU8>>,
}

impl CodePages {
    /// Zeroed straight from the allocator: one byte per page of a terabyte guest is hundreds of
    /// megabytes, and filling it through an iterator would commit all of it up front.
    pub fn new(pages: usize) -> Self {
        let zeroed = zeroed_bytes(pages).unwrap_or_else(|| {
            alloc::alloc::handle_alloc_error(Layout::array::<u8>(pages).unwrap())
        });
        let mut zeroed = core::mem::ManuallyDrop::new(zeroed);
        // `AtomicU8` has the size, alignment and bit validity of `u8`, so the allocation is
        // freed correctly as a `Vec<AtomicU8>`.
        let flags =
            unsafe { Vec::from_raw_parts(zeroed.as_mut_ptr().cast::<AtomicU8>(), pages, pages) };
        CodePages {
            flags: Arc::new(flags),
        }
    }

    pub fn len(&self) -> usize {
        self.flags.len()
    }

    pub fn get(&self, page: usize) -> u8 {
        self.flags[page].load(Ordering::Relaxed)
    }

    /// Like `get`, but a page number past the end reads as "no code".
    pub fn get_or_zero(&self, page: usize) -> u8 {
        self.flags
            .get(page)
            .map_or(0, |flag| flag.load(Ordering::Relaxed))
    }

    pub fn set(&self, page: usize, value: u8) {
        self.flags[page].store(value, Ordering::SeqCst);
    }

    /// The flags as plain bytes, for `GuestMem`'s read-only page check. Another thread may set a
    /// flag while the slice is alive; the check only needs a recent value, because
    /// `Cpu::mark_code` orders the flag against the guest bytes with a fence on its side.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.flags.as_ptr().cast::<u8>(), self.flags.len()) }
    }

    #[cfg(test)]
    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        self.flags.iter().map(|flag| flag.load(Ordering::Relaxed))
    }
}

// Indexing is for tests and the single-threaded paths that already hold a `&Cpu`/`&mut Cpu`:
// the reference points at the flag's byte, which another thread may change under it. Threaded
// code uses `get` and `set`.
impl Index<usize> for CodePages {
    type Output = u8;
    fn index(&self, page: usize) -> &u8 {
        unsafe { &*self.flags[page].as_ptr() }
    }
}

impl IndexMut<usize> for CodePages {
    fn index_mut(&mut self, page: usize) -> &mut u8 {
        unsafe { &mut *self.flags[page].as_ptr() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_of_the_ram_see_each_others_writes() {
        let mut first = GuestRam::new(vec![0; 16]);
        let second = first.clone();
        first[3] = 7;
        assert_eq!(second[3], 7);
        assert_eq!(second.len(), 16);
        assert_eq!(first.base() as usize, second.as_ptr() as usize);
    }

    #[test]
    fn the_ram_keeps_its_address_when_the_handle_moves() {
        let ram = GuestRam::new(vec![1; 4096]);
        let base = ram.base();
        let moved = Box::new(ram);
        assert_eq!(moved.base(), base);
        assert_eq!(moved[100], 1);
    }

    #[test]
    fn flat_buffer_leaves_the_pci_hole_out_of_the_ram() {
        const GIB: u64 = 1 << 30;
        assert_eq!(flat_bytes(GIB), GIB);
        assert_eq!(flat_bytes(3 * GIB), 3 * GIB);
        assert_eq!(flat_bytes(4 * GIB), 5 * GIB);
        assert_eq!(flat_bytes(3 * GIB + 1), 4 * GIB + 1);
        assert_eq!(
            (low_ram_end(5 * GIB), high_ram_end(5 * GIB)),
            (3 * GIB, 5 * GIB)
        );
        assert_eq!((low_ram_end(GIB), high_ram_end(GIB)), (GIB, 0));
        // Exactly 3 GiB has no high RAM: the buffer ends where the hole starts.
        assert_eq!(high_ram_end(3 * GIB), 0);
    }

    #[test]
    fn ranges_must_avoid_the_hole_and_the_end() {
        const GIB: u64 = 1 << 30;
        let flat = flat_bytes(4 * GIB);
        assert!(is_ram_range(flat, 0, 4096));
        assert!(is_ram_range(flat, PCI_HOLE_START - 8, 8));
        assert!(!is_ram_range(flat, PCI_HOLE_START - 4, 8));
        assert!(!is_ram_range(flat, PCI_HOLE_START, 1));
        assert!(!is_ram_range(flat, PCI_HOLE_END - 1, 1));
        assert!(!is_ram_range(flat, PCI_HOLE_END - 4, 8));
        assert!(is_ram_range(flat, PCI_HOLE_END, 8));
        assert!(is_ram_range(flat, flat - 8, 8));
        assert!(!is_ram_range(flat, flat - 4, 8));
        assert!(!is_ram_range(flat, u64::MAX, 2));
        // Without high RAM the buffer ends below the hole.
        assert!(!is_ram_range(GIB, GIB, 1));
        assert!(is_ram_range(GIB, 0, 0));
    }

    #[test]
    fn the_hole_is_not_ram_for_a_handle_either() {
        let ram = GuestRam::new(zeroed_bytes(1 << 12).unwrap());
        assert!(ram.contains(0, 1 << 12));
        assert!(!ram.contains(1 << 12, 1));
        assert_eq!(ram.slice(8, 4), Some(&[0u8; 4][..]));
        assert_eq!(ram.slice(4094, 4), None);
    }

    #[test]
    fn zeroed_bytes_reports_failure_instead_of_aborting() {
        assert_eq!(zeroed_bytes(0).unwrap().len(), 0);
        assert!(zeroed_bytes(4096).unwrap().iter().all(|&byte| byte == 0));
        assert!(zeroed_bytes(isize::MAX as usize).is_none());
    }

    #[test]
    fn views_are_checked_ranges() {
        let mut ram = GuestRam::new((0..=255).collect());
        assert_eq!(ram.len(), 256);
        assert_eq!(&ram[2..5], &[2, 3, 4]);
        assert_eq!(ram[200..].len(), 56, "an open range runs to the end");
        assert_eq!(&ram[250..][..2], &[250, 251]);
        ram[7] = 99;
        assert_eq!(ram.get(7..8), Some(&[99u8][..]));
        assert_eq!(ram.get(250..257), None);
        #[allow(clippy::reversed_empty_ranges)]
        let reversed = ram.get(9..8);
        assert_eq!(reversed, None);
        assert_eq!(ram.get(256..256), Some(&[][..]));
        assert_eq!(ram.get(256..257), None);
    }

    #[test]
    fn a_span_over_a_host_buffer_writes_through_to_it() {
        let mut bytes = vec![0u8; 8];
        let mut span = RamSpan::from(&mut bytes);
        span.reborrow()[2..4].copy_from_slice(&[5, 6]);
        span[7] = 1;
        assert_eq!(span.len(), 8);
        assert_eq!(bytes, [0, 0, 5, 6, 0, 0, 0, 1]);
    }

    #[test]
    fn rebooted_ram_is_zeroed_and_as_long() {
        let mut ram = GuestRam::zeroed(1 << 16).unwrap();
        ram[100] = 1;
        let again = ram.zeroed_again().unwrap();
        assert_eq!(again.len(), 1 << 16);
        assert!(again[0..1 << 16].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn a_duplicate_shares_nothing() {
        let mut ram = GuestRam::new(vec![3; 64]);
        let copy = ram.duplicate().unwrap();
        ram[0] = 4;
        assert_eq!(copy[0], 3);
        assert_eq!(copy.len(), 64);
        assert_ne!(copy.base(), ram.base());
    }

    #[test]
    fn views_covering_never_exceed_a_slice() {
        assert!(views_covering(10).eq(core::iter::once(0..10)));
        assert_eq!(views_covering(0).count(), 0);
        let big = MAX_VIEW_BYTES + 3;
        let pieces: Vec<_> = views_covering(big).collect();
        assert_eq!(pieces, [0..MAX_VIEW_BYTES, MAX_VIEW_BYTES..big]);
        assert_eq!(capped(5.., big), 5..big);
        assert_eq!(capped(1.., big), 1..MAX_VIEW_BYTES + 1);
        assert_eq!(capped(0.., big), 0..MAX_VIEW_BYTES);
    }

    /// The web page's 3 GiB guests: more than the allocator may hand out on wasm32. Leaks the
    /// 2 GiB it grows for the rest of the test run, which is why it is the smallest size that
    /// takes that path.
    #[cfg(target_arch = "wasm32")]
    #[test]
    fn wasm32_ram_past_isize_max_is_grown_and_addressable() {
        const LENGTH: usize = MAX_VIEW_BYTES + 1 + (64 << 10);
        let mut ram = GuestRam::zeroed(LENGTH).expect("the module's memory can grow past 2 GiB");
        assert_eq!(ram.len(), LENGTH);
        let past_two_gib = MAX_VIEW_BYTES + 1;
        ram[past_two_gib] = 0xaa;
        ram[LENGTH - 1] = 0xbb;
        assert_eq!(&ram[past_two_gib - 1..][..2], &[0, 0xaa]);
        assert_eq!(ram[LENGTH - 1], 0xbb);
        // The host address the CPU and the JIT form for it: `ram_base + physical`.
        let host = (ram.base() as usize as u64 + past_two_gib as u64) as usize as *const u8;
        assert_eq!(unsafe { *host }, 0xaa);
        assert!(ram.get(0..LENGTH).is_none(), "no slice that long");
        assert_eq!(ram[0..].len(), MAX_VIEW_BYTES);
        let base = ram.base();
        let again = ram
            .zeroed_again()
            .expect("the last handle reuses its pages");
        assert_eq!(again.base(), base);
        assert_eq!((again[past_two_gib], again[LENGTH - 1]), (0, 0));
    }

    #[test]
    fn code_page_flags_are_shared_between_clones() {
        let flags = CodePages::new(4);
        let other = flags.clone();
        flags.set(2, 1);
        assert_eq!(other.get(2), 1);
        assert_eq!(other.get(1), 0);
        assert_eq!(other.get_or_zero(99), 0);
        assert_eq!(other.len(), 4);
    }

    #[test]
    fn code_page_flags_index_like_a_slice() {
        let mut flags = CodePages::new(2);
        flags[1] = 1;
        assert_eq!(flags.get(1), 1);
        assert_eq!(flags[1], 1);
        assert!(flags.iter().any(|flag| flag != 0));
    }
}
