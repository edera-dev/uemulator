//! What `quick bootstrap` needs to prepare a guest before it boots: password hashes for the
//! accounts it creates, the answer files that drive each distro's installer unattended, and the
//! newc cpio archive that carries them, appended to the distro's initramfs.

mod alpine;
mod answers;
mod cpio;
mod crypt;
mod debian;
mod fedora;
mod targz;

pub use answers::{Answers, COMPLETION_MARKER, Distro, Payload, render};
pub use cpio::{CpioFile, S_IFDIR, S_IFLNK, S_IFMT, S_IFREG, append_to_initramfs};
pub use crypt::{SALT_LEN, salt_from_bytes, sha512_crypt};
