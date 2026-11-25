//! Saving a stopped machine to a snapshot and loading one into a freshly built machine
//! (`docs/plans/29-snapshots.md`, `docs/plans/34-incremental-snapshots.md`).
//!
//! The stream: magic, format version, (format 2) whether it is a delta, its id and its parent's,
//! the machine's shape (memory, processors, devices), guest time, every vCPU, the devices, RAM,
//! and the checksum `Writer::finish` appends. A restore checks the shape before reading any
//! state, so another machine's snapshot fails with a message that says what differs instead of
//! a tag error halfway through.
//!
//! A full snapshot carries every page that is not zero. A delta carries the pages written since
//! its parent was taken (`DirtyLog`) and the whole CPU and device state; it is applied over its
//! parent's RAM, which a restore of the chain loads first. Pages are stored as one repeated byte,
//! compressed (`snapshot::compress`), or as they are, whichever is shortest.
//!
//! The snapshot is the same whether the vCPUs ran time-shared or on threads, so it resumes in
//! either mode (the WebKit engine always time-shares, the native one uses threads):
//!
//! - Guest time is saved as one number and every clock resumes from it: the virtual clock's
//!   `time_base_ns`, and a realtime machine's `WallClock` base.
//! - `icount` and the PMU, which counts relative to it, are saved per vCPU with the PMU parked
//!   (`pmu_park`), each against the `icount` it was parked at: its own on threads, the
//!   machine's when time-shared (the inactive vCPUs' own copies are stale there). The restore
//!   unparks every PMU against the vCPU's restored `icount`, then parks the inactive ones again
//!   if the new machine time-shares, which is exactly what a switch between them does.
//! - The caches (soft TLB, decode cache, JIT code) are not saved; the new machine's are empty.

use super::*;
use crate::ram::DirtyLog;
use crate::snapshot::compress::{self, Compressor};
use crate::snapshot::{Reader, Snapshot, SnapshotError, SnapshotId, SnapshotInfo, Writer};

const MAGIC: &[u8; 8] = b"UEMUSNAP";
/// Bump when what saved state means changes without its layout changing (the per-type tags
/// catch layout changes).
const FORMAT_VERSION: u32 = 2;
/// Every non-zero page stored as it is, no id: what builds before delta snapshots wrote. Still
/// read.
const FORMAT_VERSION_RAW_PAGES: u32 = 1;
const KIND_FULL: u8 = 0;
const KIND_DELTA: u8 = 1;
const PAGE_BYTES: usize = 1 << crate::cpu::PAGE_SHIFT;
/// RAM goes in groups of this many pages, each led by a bitmap of the pages that follow: in a
/// full snapshot those that are not all zero (the rest are zero), in a delta those written since
/// the parent (the rest are the parent's). A group is one word of the `DirtyLog`.
const PAGES_PER_GROUP: usize = 64;
/// How a page of a format-2 snapshot is stored.
const PAGE_FILL: u8 = 0;
const PAGE_RAW: u8 = 1;
const PAGE_COMPRESSED: u8 = 2;
/// A compressed page is kept only if it beats a raw one by more than its own header.
const COMPRESSED_HEADER_BYTES: usize = 3;

fn error(message: impl Into<String>) -> SnapshotError {
    SnapshotError(message.into())
}

/// What a snapshot must have been taken from to load into `cpus`.
fn shape(cpus: &[Cpu]) -> String {
    format!(
        "ram={} cpus={} {}",
        cpus[0].ram.len(),
        cpus.len(),
        cpus[0].dev.lock().snapshot_shape()
    )
}

/// What to write.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SaveKind {
    /// Every page: a suspend, which outlives the snapshots before it.
    Full,
    /// A delta against the dirty log's parent when it has one, else full: a checkpoint.
    Checkpoint,
}

/// Writes the stopped machine `cpus` (`active` owns the machine-wide state; `threaded` says
/// whether each vCPU kept its own `icount`) to `sink`. Flushes the disks first: once the sink is
/// finished, the snapshot and the disk images describe the same moment.
///
/// The PMUs are left parked (`pmu_park`); a machine that runs on afterwards (a checkpoint) undoes
/// that with `unpark_after_save`. The dirty log is not touched: the caller re-arms it once the
/// whole checkpoint succeeded.
pub(super) fn save(
    cpus: &mut [Cpu],
    active: usize,
    threaded: bool,
    sink: &mut dyn SnapshotSink,
    kind: SaveKind,
) -> Result<SnapshotInfo, SnapshotError> {
    // First, before anything can fail: `unpark_after_save` undoes it whatever happens next.
    if threaded {
        for cpu in cpus.iter_mut() {
            cpu.pmu_park();
        }
    } else {
        cpus[active].pmu_park();
        // The inactive vCPUs were parked against the machine's `icount`, which lives here.
        let machine_icount = cpus[active].icount;
        for (index, cpu) in cpus.iter_mut().enumerate() {
            if index != active {
                cpu.icount = machine_icount;
            }
        }
    }
    #[cfg(not(freestanding))]
    if cpus.iter().any(|cpu| cpu.bios.is_some()) {
        return Err(error("a BIOS disk boot keeps BIOS state on the host"));
    }
    {
        let mut devices = cpus[active].dev.lock();
        if let Some(reason) = devices.snapshot_blocker() {
            return Err(error(reason));
        }
        devices.flush_storage().map_err(error)?;
    }
    let guest_ns = cpus[active].now_ns();
    let log = cpus[active].page_has_code.dirty_log();
    let parent = match kind {
        SaveKind::Full => None,
        SaveKind::Checkpoint => log.parent().map(SnapshotId),
    };
    let info = SnapshotInfo {
        version: FORMAT_VERSION,
        id: new_id(cpus, active, guest_ns, parent),
        parent,
    };
    let shape = shape(cpus);
    let mut w = Writer::new(sink);
    w.bytes(MAGIC)?;
    w.u32(FORMAT_VERSION)?;
    w.u8(if parent.is_some() {
        KIND_DELTA
    } else {
        KIND_FULL
    })?;
    w.bytes(&info.id.0)?;
    w.bytes(&parent.unwrap_or_default().0)?;
    shape.save(&mut w)?;
    w.u64(guest_ns)?;
    w.len(cpus.len())?;
    // Machine-wide `Cpu` fields that are saved (the generator behind rdrand) come from the vCPU
    // that holds them: index order, with the active one's values in slot 0's place.
    let rng_state = cpus[active].rng_state;
    for cpu in cpus.iter() {
        cpu.save(&mut w)?;
    }
    w.u64(rng_state)?;
    cpus[active].dev.lock().save(&mut w)?;
    let dirty = parent.is_some().then_some(log);
    save_ram(&cpus[active].ram, dirty, &mut w)?;
    w.finish()?;
    Ok(info)
}

/// Undoes `save`'s parking for a machine that runs on: what `save` parked was parked at the
/// `icount` it still has, so this moves nothing. Time-shared, only the active vCPU was parked by
/// `save`; the others were parked by the scheduler and stay so.
pub(super) fn unpark_after_save(cpus: &mut [Cpu], active: usize, threaded: bool) {
    if threaded {
        for cpu in cpus.iter_mut() {
            cpu.pmu_unpark();
        }
    } else {
        cpus[active].pmu_unpark();
    }
}

/// A fresh id: the guest's position (time, instructions, its random generator), the parent, and
/// the host's time where it has one. Two branches taken from one checkpoint on the virtual clock
/// can reach the same guest state; the host time keeps their ids apart.
fn new_id(cpus: &[Cpu], active: usize, guest_ns: u64, parent: Option<SnapshotId>) -> SnapshotId {
    /// Any fixed start; these are the bytes of "uemusnap".
    const ID_SEED: u64 = 0x7565_6d75_736e_6170;
    let mut state = ID_SEED;
    let mut mix = |value: u64| {
        state = splitmix64(state ^ value);
    };
    mix(guest_ns);
    mix(cpus[active].rng_state);
    for cpu in cpus {
        mix(cpu.icount);
        mix(cpu.rip);
    }
    if let Some(parent) = parent {
        mix(u64::from_le_bytes(parent.0[..8].try_into().unwrap()));
        mix(u64::from_le_bytes(parent.0[8..].try_into().unwrap()));
    }
    if let Some(wall) = &cpus[active].wall_clock {
        mix(wall.elapsed_ns());
    }
    #[cfg(not(freestanding))]
    if let Ok(since_epoch) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        mix(since_epoch.as_nanos() as u64);
    }
    mix(0);
    let low = state;
    let high = splitmix64(low);
    let mut id = [0; 16];
    id[..8].copy_from_slice(&low.to_le_bytes());
    id[8..].copy_from_slice(&high.to_le_bytes());
    SnapshotId(id)
}

/// SplitMix64's output step (Steele, Lea and Flood, "Fast splittable pseudorandom number
/// generators", 2014): its constants are the published ones.
fn splitmix64(mut value: u64) -> u64 {
    const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
    const MIX_1: u64 = 0xbf58_476d_1ce4_e5b9;
    const MIX_2: u64 = 0x94d0_49bb_1331_11eb;
    value = value.wrapping_add(GOLDEN_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(MIX_1);
    value = (value ^ (value >> 27)).wrapping_mul(MIX_2);
    value ^ (value >> 31)
}

/// Writes RAM: with `dirty`, a delta of the pages it marks; without, every non-zero page.
fn save_ram(
    ram: &GuestRam,
    dirty: Option<&DirtyLog>,
    w: &mut Writer<'_>,
) -> Result<(), SnapshotError> {
    let length = ram.len();
    w.len(length)?;
    let pages = length / PAGE_BYTES;
    let mut compressor = Compressor::new();
    let mut packed = Vec::with_capacity(PAGE_BYTES);
    let mut group_start = 0;
    while group_start < pages {
        let group_end = (group_start + PAGES_PER_GROUP).min(pages);
        let in_group = group_end - group_start;
        let all = if in_group == PAGES_PER_GROUP {
            u64::MAX
        } else {
            (1 << in_group) - 1
        };
        let present = match dirty {
            Some(log) => log.word(group_start / PAGES_PER_GROUP) & all,
            None => {
                let mut present = 0u64;
                for page in group_start..group_end {
                    if !is_filled_with(page_bytes(ram, page), 0) {
                        present |= 1 << (page - group_start);
                    }
                }
                present
            }
        };
        w.u64(present)?;
        let mut left = present;
        while left != 0 {
            let page = group_start + left.trailing_zeros() as usize;
            left &= left - 1;
            save_page(page_bytes(ram, page), &mut compressor, &mut packed, w)?;
        }
        group_start = group_end;
    }
    // RAM is a whole number of pages (`memory_mib`), so there is no tail.
    debug_assert_eq!(length % PAGE_BYTES, 0);
    Ok(())
}

fn save_page(
    bytes: &[u8],
    compressor: &mut Compressor,
    packed: &mut Vec<u8>,
    w: &mut Writer<'_>,
) -> Result<(), SnapshotError> {
    if is_filled_with(bytes, bytes[0]) {
        w.u8(PAGE_FILL)?;
        return w.u8(bytes[0]);
    }
    if compressor.compress(bytes, packed, PAGE_BYTES - COMPRESSED_HEADER_BYTES) {
        w.u8(PAGE_COMPRESSED)?;
        w.u16(packed.len() as u16)?;
        return w.bytes(packed);
    }
    w.u8(PAGE_RAW)?;
    w.bytes(bytes)
}

fn page_bytes(ram: &GuestRam, page: usize) -> &[u8] {
    let start = page * PAGE_BYTES;
    &ram[start..start + PAGE_BYTES]
}

fn is_filled_with(bytes: &[u8], value: u8) -> bool {
    // Eight bytes at a time: this runs over all of RAM on every full snapshot.
    let splat = u64::from_ne_bytes([value; 8]);
    let (words, tail) = bytes.as_chunks::<8>();
    words.iter().all(|word| u64::from_ne_bytes(*word) == splat)
        && tail.iter().all(|&byte| byte == value)
}

/// The header fields before the machine's shape.
struct Header {
    info: SnapshotInfo,
    delta: bool,
}

fn read_header(r: &mut Reader<'_>) -> Result<Header, SnapshotError> {
    let mut magic = [0; 8];
    r.bytes(&mut magic)?;
    if &magic != MAGIC {
        return Err(error("not a uemulator snapshot"));
    }
    let version = r.u32()?;
    match version {
        FORMAT_VERSION_RAW_PAGES => Ok(Header {
            info: SnapshotInfo {
                version,
                id: SnapshotId::default(),
                parent: None,
            },
            delta: false,
        }),
        FORMAT_VERSION => {
            let delta = match r.u8()? {
                KIND_FULL => false,
                KIND_DELTA => true,
                other => return Err(error(format!("unknown snapshot kind {other}"))),
            };
            let mut id = SnapshotId::default();
            r.bytes(&mut id.0)?;
            let mut parent = SnapshotId::default();
            r.bytes(&mut parent.0)?;
            Ok(Header {
                info: SnapshotInfo {
                    version,
                    id,
                    parent: delta.then_some(parent),
                },
                delta,
            })
        }
        _ => Err(error(format!(
            "format {version}; this build reads {FORMAT_VERSION_RAW_PAGES} and {FORMAT_VERSION}"
        ))),
    }
}

/// What `source`'s header says, without reading the rest or checking it. For a host that keeps
/// checkpoints: a delta's `parent` is the id of the snapshot to restore before it.
pub fn read_snapshot_info(source: &mut dyn SnapshotSource) -> Result<SnapshotInfo, SnapshotError> {
    let mut r = Reader::new(source);
    Ok(read_header(&mut r)?.info)
}

/// Loads a snapshot chain into `cpus`, a machine just built from the configuration the snapshots
/// were taken with, before anything has run; `threaded` says whether it will run one thread per
/// vCPU. `sources` is a full snapshot followed by deltas, each the child of the one before. On
/// error the machine is half restored and must not run.
///
/// Returns the last snapshot's header, and whether the machine may take deltas against it: every
/// snapshot of the chain is kept (`SnapshotSource::retained`) and has an id.
pub(super) fn restore(
    cpus: &mut [Cpu],
    threaded: bool,
    sources: &mut [Box<dyn SnapshotSource>],
) -> Result<(SnapshotInfo, bool), SnapshotError> {
    let mut previous: Option<SnapshotInfo> = None;
    let mut state = None;
    let mut parent_capable = true;
    for (index, source) in sources.iter_mut().enumerate() {
        let mut r = Reader::new(&mut **source);
        let header = read_header(&mut r)?;
        match (previous, header.info.parent) {
            (None, None) => {}
            (None, Some(parent)) => {
                return Err(error(format!(
                    "a delta of {parent}, which must be restored before it"
                )));
            }
            (Some(_), None) => {
                return Err(error(format!(
                    "snapshot {} of the chain is a full snapshot, not a delta",
                    index + 1
                )));
            }
            (Some(before), Some(parent)) if parent != before.id => {
                return Err(error(format!(
                    "snapshot {} of the chain is a delta of {parent}, not of {}",
                    index + 1,
                    before.id
                )));
            }
            (Some(_), Some(_)) => {}
        }
        let saved_shape = String::load_new(&mut r)?;
        let shape = shape(cpus);
        if saved_shape != shape {
            return Err(error(format!(
                "taken from another machine ({saved_shape}); this one is {shape}"
            )));
        }
        let guest_ns = r.u64()?;
        let count = r.len()?;
        if count != cpus.len() {
            return Err(error("the processor count differs"));
        }
        for cpu in cpus.iter_mut() {
            cpu.load(&mut r)?;
        }
        let rng_state = r.u64()?;
        cpus[0].dev.lock().load(&mut r)?;
        load_ram(&mut cpus[0].ram, &mut r, header.info.version, header.delta)?;
        // Each snapshot is checked whole before the next is laid over it.
        r.finish()?;
        parent_capable &= source.retained() && header.info.version >= FORMAT_VERSION;
        previous = Some(header.info);
        state = Some((guest_ns, rng_state));
    }
    let (Some(info), Some((guest_ns, rng_state))) = (previous, state) else {
        return Err(error("no snapshot to resume from"));
    };
    settle(cpus, threaded, guest_ns, rng_state);
    for source in sources.iter_mut() {
        source.finish()?;
    }
    Ok((info, parent_capable))
}

fn load_ram(
    ram: &mut GuestRam,
    r: &mut Reader<'_>,
    version: u32,
    delta: bool,
) -> Result<(), SnapshotError> {
    let length = r.len()?;
    if length != ram.len() {
        return Err(error("the memory size differs"));
    }
    let pages = length / PAGE_BYTES;
    let mut packed = vec![0; PAGE_BYTES];
    let mut group_start = 0;
    while group_start < pages {
        let group_end = (group_start + PAGES_PER_GROUP).min(pages);
        let present = r.u64()?;
        if group_end - group_start < PAGES_PER_GROUP && present >> (group_end - group_start) != 0 {
            return Err(error("a RAM bitmap names pages past the end"));
        }
        for page in group_start..group_end {
            let start = page * PAGE_BYTES;
            let bytes = &mut ram[start..start + PAGE_BYTES];
            if present & (1 << (page - group_start)) != 0 {
                if version == FORMAT_VERSION_RAW_PAGES {
                    r.bytes(bytes)?;
                } else {
                    load_page(bytes, &mut packed, r)?;
                }
            } else if !delta {
                // The boot loader left the kernel and firmware in the new machine's RAM.
                bytes.fill(0);
            }
            // A delta leaves the page as its parent restored it.
        }
        group_start = group_end;
    }
    Ok(())
}

fn load_page(bytes: &mut [u8], packed: &mut [u8], r: &mut Reader<'_>) -> Result<(), SnapshotError> {
    match r.u8()? {
        PAGE_FILL => {
            let value = r.u8()?;
            bytes.fill(value);
        }
        PAGE_RAW => r.bytes(bytes)?,
        PAGE_COMPRESSED => {
            let length = r.u16()? as usize;
            let packed = packed
                .get_mut(..length)
                .ok_or_else(|| error("a compressed page is longer than a page"))?;
            r.bytes(packed)?;
            compress::decompress(packed, bytes)
                .map_err(|_| error("the snapshot is damaged (a compressed page)"))?;
        }
        other => return Err(error(format!("unknown page encoding {other}"))),
    }
    Ok(())
}

/// Makes what was not saved agree with what was: the caches start empty, every clock continues
/// from `guest_ns`, and the PMUs are re-anchored (see the module comment).
fn settle(cpus: &mut [Cpu], threaded: bool, guest_ns: u64, rng_state: u64) {
    let time_shared = cpus.len() > 1 && !threaded;
    for cpu in cpus.iter_mut() {
        let cpl = cpu.cpl;
        cpu.set_cpl(cpl);
        cpu.flush_tlb();
        cpu.shadow_stack_on = cpu.shadow_stack_active();
        cpu.time_base_ns = guest_ns;
        cpu.slice_base_icount = cpu.icount;
        cpu.time_carry = 0;
        cpu.time_divisor = 1;
        cpu.idle_skipped_ns = 0;
        cpu.pause_resume_event = 0;
        // Due at once: the first event check recomputes every deadline from the devices.
        cpu.next_event = cpu.icount;
        cpu.pmu_unpark();
        if let Some(wall) = &cpu.wall_clock {
            wall.resume_at(guest_ns);
        }
    }
    cpus[0].rng_state = rng_state;
    if time_shared {
        // The machine's `icount` is vCPU 0's now; the others wait parked against it.
        let machine_icount = cpus[0].icount;
        for cpu in cpus[1..].iter_mut() {
            cpu.icount = machine_icount;
            cpu.slice_base_icount = machine_icount;
            cpu.pmu_park();
        }
    }
    cpus[0].dev.lock().restored();
}

/// Starts a new generation of the dirty log relative to snapshot `parent`: every page clean, and
/// no vCPU holding a write tag that would let a store skip marking one. Every vCPU of the machine
/// must be in `cpus` and stopped.
pub(super) fn arm_dirty_log(cpus: &mut [Cpu], parent: SnapshotId) {
    cpus[0].page_has_code.dirty_log().arm(parent.0);
    for cpu in cpus.iter_mut() {
        cpu.revoke_all_write_tags();
    }
}

/// For `MachineConfig::check_dirty_tracking`: the pages whose bytes differ from `copy`, taken
/// when the log was armed, without the log marking them. Any is a store path that does not mark.
pub(super) fn unmarked_changes(ram: &GuestRam, copy: &GuestRam, log: &DirtyLog) -> Vec<usize> {
    (0..ram.len() / PAGE_BYTES)
        .filter(|&page| !log.is_dirty(page) && page_bytes(ram, page) != page_bytes(copy, page))
        .collect()
}

#[cfg(test)]
mod tests;
