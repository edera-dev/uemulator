//! What `quick create --bootstrap` needs to prepare a guest before it boots: password hashes for
//! the accounts it creates and newc cpio archives appended to the distro's initramfs.

mod cpio;
mod crypt;

pub use cpio::{CpioFile, S_IFDIR, S_IFLNK, S_IFREG, append_to_initramfs};
pub use crypt::{SALT_LEN, salt_from_bytes, sha512_crypt};
