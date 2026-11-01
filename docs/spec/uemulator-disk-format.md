# The uemulator disk format (udisk), version 1

A udisk file holds a virtual disk: a fixed number of bytes the guest sees as a block device. It
is sparse (regions never written take no space), it can sit on top of another disk image as a
copy-on-write overlay, and it is laid out so that a reader can serve it from HTTP range requests
without downloading the whole file. File extension `.udisk`.

It plays the part QCOW2 plays for QEMU, with these differences:

- **One flat map, read lazily.** There is no L1/L2 tree and no refcount table. The map is one
  array of 8-byte entries, one per cluster, right after the header. A reader fetches the 4 KiB
  page of the map it needs when it needs it, so opening an image over the network costs one
  request for the header, and every later read costs at most one more for its map page.
- **Sub-cluster allocation.** Each cluster is split into 16 sub-blocks, and every sub-block has
  its own state. A 4 KiB guest write into an overlay copies at most one 4 KiB sub-block from the
  base, never a whole cluster, which matters when the base is across a network.
- **Zero is not the same as absent.** A sub-block can be "zero" (reads as zeros in this image,
  whatever the base holds) or "unallocated" (reads from the base). An overlay can zero a region
  of its base without storing anything.
- **The base is identified, not just named.** An overlay records its base's 128-bit image id and
  size, and readers refuse a base that does not match. QCOW2 trusts the file name.
- **Crash consistency by write ordering, without a journal.** Clusters are only ever appended and
  never freed, and the map is written only after the data it points to is durable (below), so
  a crash can leak space but never makes the map point at data that is not there.
- **No snapshots, compression, encryption or refcounts.** Those are what make QCOW2 complicated.
  Compaction is a copy (`uemulator disk convert`), which also lays a streamed image out in guest
  order.

## Conventions

All integers are little-endian. Offsets are in bytes from the start of the file. "Must" rules a
reader checks; an image that breaks one is corrupt, and a reader reports an error instead of
guessing. A reader must not trust any value from the file without checking its range first.

## Layout

| region | where | size |
|---|---|---|
| header | offset 0 | 4096 bytes |
| map | `map_offset` (4096-aligned, at or after 4096) | `map_entries` x 8 bytes |
| data clusters | at multiples of the cluster size, after the map | `cluster_size` each |

The file must be at least `map_offset + map_entries * 8` bytes long: the map is part of the file
from creation, all zeros at first (a writer extends the file over it, which most file systems keep
sparse). The first cluster that may hold data, `first_data_cluster`, is
`ceil((map_offset + map_entries * 8) / cluster_size)`.

## Header

| offset | size | field |
|---|---|---|
| 0x000 | 8 | `magic`: the ASCII bytes `UEMUDISK` |
| 0x008 | 4 | `version`: 1 |
| 0x00c | 4 | `header_size`: 4096 |
| 0x010 | 8 | `incompatible_features` |
| 0x018 | 8 | `compatible_features` |
| 0x020 | 8 | `virtual_size`: the disk's size in bytes, a nonzero multiple of 512 |
| 0x028 | 1 | `cluster_shift`: log2 of the cluster size, 16 to 24 (64 KiB to 16 MiB) |
| 0x029 | 7 | reserved, zero |
| 0x030 | 8 | `map_offset` |
| 0x038 | 8 | `map_entries`: `ceil(virtual_size / cluster_size)`, at most 2^32 |
| 0x040 | 16 | `image_id`: random bytes chosen when the image is created |
| 0x050 | 16 | `backing_id`: the base's `image_id`, or all zeros when the base has none (a raw image) |
| 0x060 | 8 | `backing_size`: the base's size in bytes |
| 0x068 | 2 | `backing_locator_length`: bytes of `backing_locator`, at most 3964 |
| 0x06a | 22 | reserved, zero |
| 0x080 | n | `backing_locator`: UTF-8, where to find the base (below) |
| 0xffc | 4 | `header_crc`: CRC-32 (the zlib/IEEE 802.3 polynomial) of bytes 0x000 to 0xffb |

Bytes between the end of the locator and 0xffc are zero. A version 1 reader rejects a header whose
`version` is not 1, whose `header_size` is not 4096 or whose CRC does not match.

The header is written once, when the image is created, and never changes afterwards. A writer
that needs to change it (a future version) must write a new file.

### Feature bits

A reader refuses an image with an `incompatible_features` bit it does not know, because reading it
without that knowledge would return wrong data. It ignores unknown `compatible_features` bits.

| bit | set | name | meaning |
|---|---|---|---|
| 0 | incompatible | `BACKING` | The image is an overlay. Unallocated sub-blocks read from the base, and `backing_size` is nonzero. Without the bit unallocated sub-blocks read as zeros and the `backing_*` fields are zero. |

No compatible feature is defined in version 1.

### The base

`backing_locator` says where the base is, for tools that open the overlay by itself: a path
(relative paths are relative to the directory of the overlay file) or an `http://` or `https://`
URL. It may be empty when whoever opens the overlay supplies the base itself, as the web page does
for an in-memory overlay. The base may be a raw image or another udisk image, which may in turn
have a base of its own.

When the overlay is opened, the base must be exactly `backing_size` bytes long, and if
`backing_id` is not all zeros, the base must be a udisk image whose `image_id` equals it. A
mismatch is an error, not a warning: an overlay over a different base reads as a mix of two disks.
The overlay's `virtual_size` may differ from `backing_size`; sub-blocks past the end of the base
read as zeros.

A base must not change while an overlay depends on it. Tools open bases read-only, but nothing in
the format can stop another program from writing one; the image id catches a base that was
replaced or reconverted, not one that was modified in place.

## Clusters and sub-blocks

`cluster_size` is `1 << cluster_shift`. Cluster `n` of the virtual disk covers bytes
`n * cluster_size` to `(n + 1) * cluster_size - 1`; the last cluster may extend past
`virtual_size`, and that part is never read or written.

Every cluster is split into 16 sub-blocks of `cluster_size / 16` bytes (4 KiB with the default
64 KiB clusters). Sub-block `i` of a cluster covers its bytes `i * sub_block_size` to
`(i + 1) * sub_block_size - 1`.

## The map

Entry `n` of the map, at `map_offset + n * 8`, describes cluster `n` of the virtual disk:

| bits | field |
|---|---|
| 0 to 31 | `data_cluster`: where this cluster's data is stored in the file, in units of `cluster_size` (file offset `data_cluster << cluster_shift`); 0 when no space is allocated |
| 32 to 47 | `data_mask`: bit `i` set means sub-block `i` is stored in the file, at `(data_cluster << cluster_shift) + i * sub_block_size` |
| 48 to 63 | `zero_mask`: bit `i` set means sub-block `i` reads as zeros |

Each sub-block is in one of three states:

| `data_mask` bit | `zero_mask` bit | state | reads as |
|---|---|---|---|
| 1 | 0 | data | the bytes stored in the file |
| 0 | 1 | zero | zeros |
| 0 | 0 | unallocated | the base at the same offset with `BACKING` (zeros past its end), else zeros |
| 1 | 1 | invalid | the image is corrupt |

An all-zero entry, which is what a new map holds, is a cluster with every sub-block unallocated.

The rules a reader checks:

- No sub-block has both its bits set.
- If `data_mask` is nonzero, `data_cluster` is nonzero.
- A nonzero `data_cluster` is at least `first_data_cluster` and starts before the end of the file
  (rounded up to a whole cluster). A writer allocating at the end of the file would otherwise hand
  that space out a second time. The stored sub-blocks must lie within the file.
- Mask bits for sub-blocks that lie entirely past `virtual_size` are zero.

A `data_cluster` with an empty `data_mask` is allowed: the space was allocated and a writer may
use it for later writes to that cluster. Two entries must not name the same `data_cluster`. A
reader cannot check that cheaply and need not; a writer never produces it.

The map is read in pages of 4096 bytes (512 entries) at `map_offset + p * 4096`; the last page may
be shorter. That is only a convenience for readers: nothing in the file depends on it.

## Writing

The format allows one writer at a time and no readers while it writes. These rules are what make
a crash at any point leave a consistent image.

1. **Allocate by appending.** A cluster needs file space the first time a sub-block of it becomes
   data. The writer takes the cluster at the end of the file: `max(first_data_cluster,
   ceil(file_length / cluster_size))` when it opens the image, counting up from there. Space is
   never freed or reused.
2. **Write whole sub-blocks when a sub-block becomes data.** A write that covers part of a
   sub-block in state zero or unallocated fills in the rest from what the sub-block read as
   before (zeros or the base) and stores the full sub-block. A write to a sub-block already in
   state data changes just the bytes written.
3. **Map changes wait for a flush.** Changed map entries stay in memory. On a flush (the guest's
   flush request, or closing the image) the writer, in this order: makes the data it wrote
   durable (`fsync`), writes the changed map pages, and makes them durable.

A crash before step 3 completes loses the writes since the last flush, which a guest that did not
flush may not rely on, and may leave appended clusters that nothing points to; those cost space
until the image is copied. Map entries are 8-byte aligned and so never straddle a 512-byte
sector, so a torn page write leaves every entry either old or new, and both point at durable data.
Overwriting a sub-block that is already data is not ordered against anything, exactly like a
write to a raw disk.

A writer may also store less than it was given. A write that covers a whole sub-block in state
zero or unallocated with all zero bytes stores nothing: the sub-block becomes zero in an overlay
(the base may hold something else there) and stays unallocated in an image without a base.

## Reading over a network

Nothing in a version 1 image needs a write to read, so a read-only image can be served by any HTTP
server that honours `Range` requests (status 206). A reader needs the total length (from
`Content-Range`), then fetches:

- bytes 0 to 4095 for the header,
- a map page the first time a cluster it covers is read,
- the stored sub-blocks of each read.

Readers are expected to cache: uemulator fetches aligned 256 KiB chunks, keeps the most recently
used ones, and fetches several chunks at once when reads are sequential. `uemulator disk convert`
writes the header, then the map, then the data clusters in guest order, so that a guest reading
its disk front to back reads the file front to back.

## Checking an implementation

`uemulator disk info FILE` prints the header and how much of the disk is data, zero and
unallocated. The tests in `crates/uemulator/src/disk/udisk_tests.rs` check the layout byte by
byte, the writing rules and their order, and corrupt valid images to check that a reader refuses
every image that breaks a rule above.

## Durability

The writer keeps changed map entries in memory until a flush, so a process that is killed loses
the guest's writes since its last flush request, the way a disk with a volatile write cache loses
them on power failure (virtio-blk tells the guest it has one). Closing the image flushes.
