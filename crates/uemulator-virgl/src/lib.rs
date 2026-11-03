//! The host side of virgl, the protocol Mesa's virgl driver speaks through virtio-gpu
//! (`docs/plans/25-gpu-3d.md`). The library's device only checks and forwards; what the guest's
//! command streams mean is decided here, outside it. So far only the capsets: what the host
//! tells the guest's driver it can do. `no_std` with no dependencies, so the web page's module
//! and the CLI can both link it.
#![no_std]

extern crate alloc;

pub mod caps;
