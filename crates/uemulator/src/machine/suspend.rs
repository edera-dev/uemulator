//! Saving a stopped machine to a snapshot and loading one into a freshly built machine
//! (`docs/plans/29-snapshots.md`).
//!
//! The stream: magic, format version, the machine's shape (memory, processors, devices), guest
//! time, every vCPU, the devices, RAM, and the checksum `Writer::finish` appends. A restore
//! checks the shape before reading any state, so another machine's snapshot fails with a message
//! that says what differs instead of a tag error halfway through.
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
use crate::snapshot::{Reader, Snapshot, SnapshotError, Writer};

const MAGIC: &[u8; 8] = b"UEMUSNAP";
/// Bump when what saved state means changes without its layout changing (the per-type tags
/// catch layout changes).
const FORMAT_VERSION: u32 = 1;
const PAGE_BYTES: usize = 1 << crate::cpu::PAGE_SHIFT;
/// RAM goes in groups of this many pages, each led by a bitmap of the ones that are not all
/// zero: those follow, the rest are skipped. A guest rarely touches most of its RAM, and the
/// WebKit engine sends every byte through a POST.
const PAGES_PER_GROUP: usize = 64;

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

/// Writes the stopped machine `cpus` (`active` owns the machine-wide state; `threaded` says
/// whether each vCPU kept its own `icount`) to `sink`. Flushes the disks first: once the sink is
/// finished, the snapshot and the disk images describe the same moment.
pub(super) fn save(
    cpus: &mut [Cpu],
    active: usize,
    threaded: bool,
    sink: &mut dyn SnapshotSink,
) -> Result<(), SnapshotError> {
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
    let shape = shape(cpus);
    let mut w = Writer::new(sink);
    w.bytes(MAGIC)?;
    w.u32(FORMAT_VERSION)?;
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
    save_ram(&cpus[active].ram, &mut w)?;
    w.finish()
}

fn save_ram(ram: &GuestRam, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
    let length = ram.len();
    w.len(length)?;
    let pages = length / PAGE_BYTES;
    let mut group_start = 0;
    while group_start < pages {
        let group_end = (group_start + PAGES_PER_GROUP).min(pages);
        let mut present = 0u64;
        for page in group_start..group_end {
            if !is_zero(page_bytes(ram, page)) {
                present |= 1 << (page - group_start);
            }
        }
        w.u64(present)?;
        for page in group_start..group_end {
            if present & (1 << (page - group_start)) != 0 {
                w.bytes(page_bytes(ram, page))?;
            }
        }
        group_start = group_end;
    }
    // RAM is a whole number of pages (`memory_mib`), so there is no tail.
    debug_assert_eq!(length % PAGE_BYTES, 0);
    Ok(())
}

fn page_bytes(ram: &GuestRam, page: usize) -> &[u8] {
    let start = page * PAGE_BYTES;
    &ram[start..start + PAGE_BYTES]
}

fn is_zero(bytes: &[u8]) -> bool {
    // Eight bytes at a time: this runs over all of RAM on every suspend.
    let (words, tail) = bytes.as_chunks::<8>();
    words.iter().all(|word| u64::from_ne_bytes(*word) == 0) && tail.iter().all(|&byte| byte == 0)
}

/// Loads a snapshot into `cpus`, a machine just built from the configuration the snapshot was
/// taken with, before anything has run; `threaded` says whether it will run one thread per vCPU.
/// On error the machine is half restored and must not run.
pub(super) fn restore(
    cpus: &mut [Cpu],
    threaded: bool,
    source: &mut dyn SnapshotSource,
) -> Result<(), SnapshotError> {
    let mut r = Reader::new(source);
    let mut magic = [0; 8];
    r.bytes(&mut magic)?;
    if &magic != MAGIC {
        return Err(error("not a uemulator snapshot"));
    }
    let version = r.u32()?;
    if version != FORMAT_VERSION {
        return Err(error(format!(
            "format {version}; this build reads {FORMAT_VERSION}"
        )));
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
    load_ram(&mut cpus[0].ram, &mut r)?;
    r.finish()?;
    settle(cpus, threaded, guest_ns, rng_state);
    source.finish()
}

fn load_ram(ram: &mut GuestRam, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
    let length = r.len()?;
    if length != ram.len() {
        return Err(error("the memory size differs"));
    }
    let pages = length / PAGE_BYTES;
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
                r.bytes(bytes)?;
            } else {
                // The boot loader left the kernel and firmware in the new machine's RAM.
                bytes.fill(0);
            }
        }
        group_start = group_end;
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

#[cfg(test)]
mod tests;
