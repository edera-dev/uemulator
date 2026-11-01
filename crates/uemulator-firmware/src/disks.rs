//! Disks as UEFI sees them: a handle per virtio disk, per partition and per El Torito EFI boot
//! image, each with a device path, `BlockIO` and `DiskIO`, and a `SimpleFileSystem` where the
//! contents are FAT.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::ptr;

use crate::devicepath::{self, DevicePath};
use crate::eltorito;
use crate::fat::{self, FatError};
use crate::files;
use crate::gpt::{self, Blocks, Partition, Scheme};
use crate::protocols;
use crate::sync::Global;
use crate::uefi::*;
use crate::virtio_blk::{self, SECTOR_BYTES, VirtioBlk};

const BLOCK_IO_REVISION: u64 = 0x0002_0001;
const DISK_IO_REVISION: u64 = 0x0001_0000;
const MEDIA_ID: u32 = 0;

#[repr(C)]
struct BlockMedia {
    media_id: u32,
    removable_media: u8,
    media_present: u8,
    logical_partition: u8,
    read_only: u8,
    write_caching: u8,
    block_size: u32,
    io_align: u32,
    last_block: u64,
    lowest_aligned_lba: u64,
    logical_blocks_per_physical_block: u32,
    optimal_transfer_length_granularity: u32,
}

#[repr(C)]
struct BlockDevice {
    revision: u64,
    media: *mut BlockMedia,
    reset: extern "efiapi" fn(*mut BlockDevice, u8) -> Status,
    read_blocks: extern "efiapi" fn(*mut BlockDevice, u32, u64, usize, *mut c_void) -> Status,
    write_blocks: extern "efiapi" fn(*mut BlockDevice, u32, u64, usize, *const c_void) -> Status,
    flush_blocks: extern "efiapi" fn(*mut BlockDevice) -> Status,
    media_data: BlockMedia,
    disk: *mut VirtioBlk,
    first_block: u64,
    block_count: u64,
}

#[repr(C)]
struct DiskIo {
    revision: u64,
    read_disk: extern "efiapi" fn(*mut DiskIo, u32, u64, usize, *mut c_void) -> Status,
    write_disk: extern "efiapi" fn(*mut DiskIo, u32, u64, usize, *const c_void) -> Status,
    device: *mut BlockDevice,
}

static DISKS: Global<Vec<*mut VirtioBlk>> = Global::new(Vec::new());

/// Byte-addressed access to a `BlockDevice`, for the FAT reader and the partition scanner.
pub struct BlockStorage {
    device: *mut BlockDevice,
}

fn read_bytes(device: &mut BlockDevice, offset: u64, buffer: &mut [u8]) -> bool {
    let block = SECTOR_BYTES as u64;
    let end = offset + buffer.len() as u64;
    if end > device.block_count * block {
        return false;
    }
    let disk = unsafe { &mut *device.disk };
    if offset.is_multiple_of(block) && (buffer.len() as u64).is_multiple_of(block) {
        return disk.read(device.first_block + offset / block, buffer);
    }
    let first = offset / block;
    let mut bounce = alloc::vec![0u8; ((end.div_ceil(block) - first) * block) as usize];
    if !disk.read(device.first_block + first, &mut bounce) {
        return false;
    }
    let skip = (offset - first * block) as usize;
    buffer.copy_from_slice(&bounce[skip..skip + buffer.len()]);
    true
}

impl fat::Storage for BlockStorage {
    fn read(&mut self, offset: u64, buffer: &mut [u8]) -> Result<(), FatError> {
        read_bytes(unsafe { &mut *self.device }, offset, buffer)
            .then_some(())
            .ok_or(FatError::Device)
    }
}

impl Blocks for BlockStorage {
    fn block_bytes(&self) -> u64 {
        SECTOR_BYTES as u64
    }

    fn block_count(&self) -> u64 {
        unsafe { (*self.device).block_count }
    }

    fn read_blocks(&mut self, first: u64, buffer: &mut [u8]) -> bool {
        read_bytes(
            unsafe { &mut *self.device },
            first * SECTOR_BYTES as u64,
            buffer,
        )
    }
}

extern "efiapi" fn reset(_: *mut BlockDevice, _: u8) -> Status {
    SUCCESS
}

/// `Err` carries the status to return, `SUCCESS` for a transfer of nothing.
fn check_transfer(
    device: &BlockDevice,
    media_id: u32,
    lba: u64,
    size: usize,
    buffer: *const c_void,
) -> Result<(), Status> {
    if media_id != MEDIA_ID {
        return Err(MEDIA_CHANGED);
    }
    if !size.is_multiple_of(SECTOR_BYTES) {
        return Err(BAD_BUFFER_SIZE);
    }
    if size == 0 {
        return Err(SUCCESS);
    }
    if buffer.is_null()
        || lba >= device.block_count
        || (size / SECTOR_BYTES) as u64 > device.block_count - lba
    {
        return Err(INVALID_PARAMETER);
    }
    Ok(())
}

extern "efiapi" fn read_blocks(
    this: *mut BlockDevice,
    media_id: u32,
    lba: u64,
    size: usize,
    buffer: *mut c_void,
) -> Status {
    let device = unsafe { &mut *this };
    if let Err(status) = check_transfer(device, media_id, lba, size, buffer) {
        return status;
    }
    let destination = unsafe { core::slice::from_raw_parts_mut(buffer.cast::<u8>(), size) };
    if unsafe { &mut *device.disk }.read(device.first_block + lba, destination) {
        SUCCESS
    } else {
        DEVICE_ERROR
    }
}

extern "efiapi" fn write_blocks(
    this: *mut BlockDevice,
    media_id: u32,
    lba: u64,
    size: usize,
    buffer: *const c_void,
) -> Status {
    let device = unsafe { &mut *this };
    if device.media_data.read_only != 0 {
        return WRITE_PROTECTED;
    }
    if let Err(status) = check_transfer(device, media_id, lba, size, buffer) {
        return status;
    }
    let source = unsafe { core::slice::from_raw_parts(buffer.cast::<u8>(), size) };
    if unsafe { &mut *device.disk }.write(device.first_block + lba, source) {
        SUCCESS
    } else {
        DEVICE_ERROR
    }
}

extern "efiapi" fn flush_blocks(this: *mut BlockDevice) -> Status {
    if unsafe { &mut *(*this).disk }.flush() {
        SUCCESS
    } else {
        DEVICE_ERROR
    }
}

extern "efiapi" fn read_disk(
    this: *mut DiskIo,
    media_id: u32,
    offset: u64,
    size: usize,
    buffer: *mut c_void,
) -> Status {
    let device = unsafe { &mut *(*this).device };
    if media_id != MEDIA_ID {
        return MEDIA_CHANGED;
    }
    if size == 0 {
        return SUCCESS;
    }
    if buffer.is_null() {
        return INVALID_PARAMETER;
    }
    let destination = unsafe { core::slice::from_raw_parts_mut(buffer.cast::<u8>(), size) };
    if read_bytes(device, offset, destination) {
        SUCCESS
    } else {
        DEVICE_ERROR
    }
}

extern "efiapi" fn write_disk(
    this: *mut DiskIo,
    media_id: u32,
    offset: u64,
    size: usize,
    buffer: *const c_void,
) -> Status {
    let device = unsafe { &mut *(*this).device };
    if media_id != MEDIA_ID {
        return MEDIA_CHANGED;
    }
    let block = SECTOR_BYTES as u64;
    if device.media_data.read_only != 0 {
        return WRITE_PROTECTED;
    }
    if size == 0 {
        return SUCCESS;
    }
    if buffer.is_null() || offset + size as u64 > device.block_count * block {
        return INVALID_PARAMETER;
    }
    // Read-modify-write the blocks the range touches.
    let first = offset / block;
    let length = ((offset + size as u64).div_ceil(block) - first) * block;
    let mut bounce = alloc::vec![0u8; length as usize];
    let disk = unsafe { &mut *device.disk };
    if !disk.read(device.first_block + first, &mut bounce) {
        return DEVICE_ERROR;
    }
    let skip = (offset - first * block) as usize;
    bounce[skip..skip + size]
        .copy_from_slice(unsafe { core::slice::from_raw_parts(buffer.cast::<u8>(), size) });
    if disk.write(device.first_block + first, &bounce) {
        SUCCESS
    } else {
        DEVICE_ERROR
    }
}

fn new_device(
    disk: *mut VirtioBlk,
    first_block: u64,
    block_count: u64,
    logical_partition: bool,
) -> *mut BlockDevice {
    let read_only = unsafe { (*disk).read_only } as u8;
    let device = Box::into_raw(Box::new(BlockDevice {
        revision: BLOCK_IO_REVISION,
        media: ptr::null_mut(),
        reset,
        read_blocks,
        write_blocks,
        flush_blocks,
        media_data: BlockMedia {
            media_id: MEDIA_ID,
            removable_media: 0,
            media_present: 1,
            logical_partition: logical_partition as u8,
            read_only,
            write_caching: 0,
            block_size: SECTOR_BYTES as u32,
            io_align: 0,
            last_block: block_count - 1,
            lowest_aligned_lba: 0,
            logical_blocks_per_physical_block: 1,
            optimal_transfer_length_granularity: 0,
        },
        disk,
        first_block,
        block_count,
    }));
    unsafe { (*device).media = &raw mut (*device).media_data };
    device
}

fn leak_path(path: &DevicePath) -> *mut c_void {
    Box::leak(path.as_bytes().to_vec().into_boxed_slice())
        .as_mut_ptr()
        .cast()
}

/// Publishes one block device: handle, path, `BlockIO`, `DiskIO` and, if it holds FAT, a file
/// system. Returns the handle.
fn publish(device: *mut BlockDevice, path: &DevicePath, with_file_system: bool) -> Option<Handle> {
    let handle = protocols::install(ptr::null_mut(), guids::DEVICE_PATH, leak_path(path)).ok()?;
    let _ = protocols::install(handle, guids::BLOCK_IO, device.cast());
    let disk_io = Box::into_raw(Box::new(DiskIo {
        revision: DISK_IO_REVISION,
        read_disk,
        write_disk,
        device,
    }));
    let _ = protocols::install(handle, guids::DISK_IO, disk_io.cast());
    let volume = if with_file_system {
        files::create_volume(BlockStorage { device })
    } else {
        None
    };
    crate::serial::trace!("  {}: file system {}", path.text(), volume.is_some());
    if let Some(volume) = volume {
        let _ = protocols::install(handle, guids::SIMPLE_FILE_SYSTEM, volume.cast());
    }
    Some(handle)
}

fn partition_path(disk_path: &DevicePath, partition: &Partition) -> DevicePath {
    let (signature, format, signature_type) = match &partition.scheme {
        Scheme::Gpt { unique_guid, .. } => (
            unique_guid.to_bytes(),
            devicepath::PARTITION_FORMAT_GPT,
            devicepath::SIGNATURE_GUID,
        ),
        Scheme::Mbr { disk_signature, .. } => {
            let mut signature = [0u8; 16];
            signature[..4].copy_from_slice(&disk_signature.to_le_bytes());
            (
                signature,
                devicepath::PARTITION_FORMAT_MBR,
                devicepath::SIGNATURE_MBR,
            )
        }
    };
    disk_path.clone().hard_drive(
        partition.number,
        partition.first_block,
        partition.block_count,
        signature,
        format,
        signature_type,
    )
}

pub fn init() {
    for mut disk in virtio_blk::probe_all() {
        if disk.sector_count == 0 {
            continue;
        }
        let function = disk.function;
        let raw = &mut *disk as *mut VirtioBlk;
        DISKS.get().push(raw);
        core::mem::forget(disk);
        let disk_path = DevicePath::new()
            .acpi_pci_root()
            .pci(function.device, function.function);
        let whole = new_device(raw, 0, unsafe { (*raw).sector_count }, false);
        if cfg!(feature = "trace") {
            let mut sector = [0u8; 512];
            let ok = unsafe { (*raw).read(1, &mut sector) };
            crate::serial::trace!("read sector 1: {ok} {:02x?}", &sector[..16]);
        }
        let partitions = gpt::partitions(&mut BlockStorage { device: whole });
        crate::serial::trace!(
            "disk {}: {} sectors, {} partitions",
            function.device,
            unsafe { (*raw).sector_count },
            partitions.as_ref().map_or(0, |found| found.len())
        );
        // An ISO image without a hybrid partition table: its EFI boot image is found the way a
        // CD's is. Only then, as EDK2 tries GPT before El Torito: a hybrid's GPT and catalog
        // usually name the same FAT image, and two volumes would give it two boot entries.
        let boot_image = match partitions {
            None => eltorito::efi_boot_image(&mut BlockStorage { device: whole }),
            Some(_) => None,
        };
        crate::serial::trace!("  El Torito EFI image: {:?}", boot_image);
        // A disk that has partitions does not also offer a file system spanning them.
        publish(
            whole,
            &disk_path,
            partitions.is_none() && boot_image.is_none(),
        );
        for partition in partitions.unwrap_or_default() {
            let device = new_device(raw, partition.first_block, partition.block_count, true);
            publish(device, &partition_path(&disk_path, &partition), true);
        }
        if let Some(image) = boot_image {
            let device = new_device(raw, image.first_block, image.block_count, true);
            let path = disk_path
                .clone()
                .cd_rom(image.entry, image.cd_sector, image.block_count);
            publish(device, &path, true);
        }
    }
}

/// Stops every disk so the OS driver finds idle devices.
pub fn shutdown() {
    for &disk in DISKS.get().iter() {
        unsafe { (*disk).reset() };
    }
}
