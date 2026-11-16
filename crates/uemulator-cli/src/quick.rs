use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::run::DEFAULT_MEMORY_MIB;

/// What a Fedora install gets: its stage2 image (~900 MB) is downloaded into RAM and must fit
/// beside Anaconda.
const FEDORA_INSTALL_MEMORY_MIB: usize = 3072;
const DOWNLOAD_CHUNK_LEN: usize = 1 << 16;

/// The lowercase name is both the command-line value and what `machine.toml` stores. The Apple
/// app keeps a copy of these URLs, file names, command lines and memory sizes
/// (`apple/App/Presets.swift`): change both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Distro {
    Debian,
    Alpine,
    Fedora,
}

impl Distro {
    pub fn default_memory_mib(self) -> usize {
        match self {
            Distro::Fedora => FEDORA_INSTALL_MEMORY_MIB,
            _ => DEFAULT_MEMORY_MIB,
        }
    }

    /// Kernel arguments that make the netboot image reach its installer over the user-mode network.
    /// `ip=dhcp` configures the NIC in the initramfs; Alpine's `modloop` must come from the same
    /// release as the cached kernel, so `update-artifacts` refreshes both together.
    pub fn cmdline(self) -> &'static str {
        match self {
            Distro::Debian => "",
            // Alpine's virt kernel is built without `earlyprintk=`, so with the default hvc0
            // console everything before the virtio-console driver loads (including the banner)
            // would be lost; `earlycon=` is built in and reaches the same UART.
            Distro::Alpine => {
                "earlycon=uart8250,io,0x3f8 ip=dhcp alpine_repo=https://dl-cdn.alpinelinux.org/alpine/latest-stable/main modloop=https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot/modloop-virt"
            }
            Distro::Fedora => {
                "ip=dhcp rd.neednet=1 inst.text inst.repo=https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Server/x86_64/os/"
            }
        }
    }
}

pub struct Artifacts {
    pub kernel: PathBuf,
    pub initramfs: PathBuf,
}

struct Download {
    /// Name under the machine's artifacts directory.
    file_name: &'static str,
    url: &'static str,
}

struct NetbootImages {
    kernel: Download,
    initramfs: Download,
}

impl Distro {
    pub fn name(self) -> &'static str {
        match self {
            Distro::Debian => "debian",
            Distro::Alpine => "alpine",
            Distro::Fedora => "fedora",
        }
    }

    /// The kernel and initramfs URLs, for tests that check what was fetched.
    #[cfg(test)]
    pub fn source_urls(self) -> (&'static str, &'static str) {
        let source = self.source();
        (source.kernel.url, source.initramfs.url)
    }

    /// Debian and Fedora URLs are pinned to a release (Debian `trixie`, Fedora 44) and must be
    /// bumped by hand; Alpine tracks `latest-stable`.
    fn source(self) -> NetbootImages {
        match self {
            Distro::Debian => NetbootImages {
                kernel: Download {
                    file_name: "linux",
                    url: "https://deb.debian.org/debian/dists/trixie/main/installer-amd64/current/images/netboot/debian-installer/amd64/linux",
                },
                initramfs: Download {
                    file_name: "initrd.gz",
                    url: "https://deb.debian.org/debian/dists/trixie/main/installer-amd64/current/images/netboot/debian-installer/amd64/initrd.gz",
                },
            },
            // boot.alpinelinux.org only serves iPXE images; the netboot files live on the CDN.
            Distro::Alpine => NetbootImages {
                kernel: Download {
                    file_name: "vmlinuz-virt",
                    url: "https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot/vmlinuz-virt",
                },
                initramfs: Download {
                    file_name: "initramfs-virt",
                    url: "https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot/initramfs-virt",
                },
            },
            Distro::Fedora => NetbootImages {
                kernel: Download {
                    file_name: "vmlinuz",
                    url: "https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Server/x86_64/os/images/pxeboot/vmlinuz",
                },
                initramfs: Download {
                    file_name: "initrd.img",
                    url: "https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Server/x86_64/os/images/pxeboot/initrd.img",
                },
            },
        }
    }
}

/// A zero-length file counts as missing: it is what a crash between create and first write leaves.
fn needs_download(path: &Path, update: bool) -> bool {
    let cached = fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false);
    update || !cached
}

/// Whether `ensure_in` would find both of the distro's images already in `dir`.
pub fn artifacts_present(dir: &Path, distro: Distro) -> bool {
    let source = distro.source();
    [source.kernel, source.initramfs]
        .iter()
        .all(|image| !needs_download(&dir.join(image.file_name), false))
}

/// Puts the distro's kernel and initramfs in `dir`, fetching only what is missing unless `update`.
/// The transfer is injected so the skip logic runs without a network.
pub fn ensure_in(
    dir: &Path,
    distro: Distro,
    update: bool,
    mut fetch_url: impl FnMut(&str, &Path) -> Result<(), String>,
) -> Result<Artifacts, String> {
    fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let src = distro.source();
    let mut fetch = |image: Download| -> Result<PathBuf, String> {
        let path = dir.join(image.file_name);
        if needs_download(&path, update) {
            fetch_url(image.url, &path)?;
        }
        Ok(path)
    };
    Ok(Artifacts {
        kernel: fetch(src.kernel)?,
        initramfs: fetch(src.initramfs)?,
    })
}

/// Streams to a `.part` file and renames it into place, so an interrupted download never leaves a
/// truncated artifact that `ensure` would trust.
pub fn download(url: &str, dest: &Path) -> Result<(), String> {
    eprintln!("downloading {url}");
    let res = ureq::get(url)
        .call()
        .map_err(|e| io::Error::other(e.to_string()))
        .and_then(|resp| write_atomically(resp.into_body().into_reader(), dest));
    res.map_err(|e| format!("download of {url} failed: {e}"))
}

fn part_path(dest: &Path) -> PathBuf {
    dest.with_extension("part")
}

/// Copies `body` to `dest` through `part_path(dest)`. On any failure the partial file is removed
/// and `dest` is left as it was.
pub fn write_atomically(mut body: impl Read, dest: &Path) -> io::Result<()> {
    let part = part_path(dest);
    let res = (|| -> io::Result<()> {
        let mut out = File::create(&part)?;
        let mut buf = vec![0u8; DOWNLOAD_CHUNK_LEN];
        loop {
            let n = body.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
        }
        out.sync_all()?;
        fs::rename(&part, dest)
    })();
    if res.is_err() {
        let _ = fs::remove_file(&part);
    }
    res
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::Cursor;

    use super::*;
    use crate::test_support::TempDir;

    const ALL_DISTROS: [Distro; 3] = [Distro::Debian, Distro::Alpine, Distro::Fedora];

    /// The release component of `.../releases/<n>/...`.
    fn release_in(url: &str) -> &str {
        let after = url
            .split("/releases/")
            .nth(1)
            .expect("no /releases/ in url");
        after.split('/').next().unwrap()
    }

    #[test]
    fn distro_names_are_the_lowercase_command_line_names() {
        assert_eq!(Distro::Debian.name(), "debian");
        assert_eq!(Distro::Alpine.name(), "alpine");
        assert_eq!(Distro::Fedora.name(), "fedora");
    }

    #[test]
    fn only_fedora_gets_more_than_the_default_memory() {
        assert_eq!(Distro::Debian.default_memory_mib(), DEFAULT_MEMORY_MIB);
        assert_eq!(Distro::Alpine.default_memory_mib(), DEFAULT_MEMORY_MIB);
        assert_eq!(
            Distro::Fedora.default_memory_mib(),
            FEDORA_INSTALL_MEMORY_MIB
        );
        assert!(Distro::Fedora.default_memory_mib() > DEFAULT_MEMORY_MIB);
    }

    #[test]
    fn the_fedora_install_fits_the_library_limits() {
        assert!(
            (uemulator::MIN_MEMORY_MIB..=uemulator::MAX_MEMORY_MIB)
                .contains(&FEDORA_INSTALL_MEMORY_MIB)
        );
    }

    #[test]
    fn every_source_url_is_https_and_ends_in_its_cache_file_name() {
        for distro in ALL_DISTROS {
            let source = distro.source();
            for image in [&source.kernel, &source.initramfs] {
                assert!(image.url.starts_with("https://"), "{}", image.url);
                assert!(
                    image.url.ends_with(&format!("/{}", image.file_name)),
                    "{} vs {}",
                    image.url,
                    image.file_name
                );
                assert!(!image.file_name.contains('/'));
                assert!(!image.file_name.is_empty());
            }
        }
    }

    #[test]
    fn kernel_and_initramfs_have_different_names_and_urls_per_distro() {
        for distro in ALL_DISTROS {
            let source = distro.source();
            assert_ne!(source.kernel.file_name, source.initramfs.file_name);
            assert_ne!(source.kernel.url, source.initramfs.url);
        }
    }

    #[test]
    fn part_files_of_one_distro_never_collide() {
        for distro in ALL_DISTROS {
            let source = distro.source();
            let dir = Path::new("/cache");
            assert_ne!(
                part_path(&dir.join(source.kernel.file_name)),
                part_path(&dir.join(source.initramfs.file_name))
            );
        }
    }

    #[test]
    fn debian_files_match_the_netboot_layout() {
        let source = Distro::Debian.source();
        assert_eq!(source.kernel.file_name, "linux");
        assert_eq!(source.initramfs.file_name, "initrd.gz");
        assert!(source.kernel.url.contains("/dists/trixie/"));
        assert!(
            source
                .kernel
                .url
                .contains("/installer-amd64/current/images/netboot/")
        );
        assert!(source.kernel.url.starts_with("https://deb.debian.org/"));
    }

    #[test]
    fn alpine_gets_earlycon_because_its_kernel_has_no_earlyprintk() {
        assert!(
            Distro::Alpine
                .cmdline()
                .contains("earlycon=uart8250,io,0x3f8")
        );
        assert!(!Distro::Debian.cmdline().contains("earlycon"));
    }

    #[test]
    fn alpine_tracks_latest_stable_for_kernel_and_modloop_alike() {
        let source = Distro::Alpine.source();
        assert_eq!(source.kernel.file_name, "vmlinuz-virt");
        assert_eq!(source.initramfs.file_name, "initramfs-virt");
        let cmdline = Distro::Alpine.cmdline();
        for url in [source.kernel.url, source.initramfs.url] {
            assert!(
                url.contains("/alpine/latest-stable/releases/x86_64/netboot/"),
                "{url}"
            );
        }
        assert!(
            cmdline
                .contains("alpine_repo=https://dl-cdn.alpinelinux.org/alpine/latest-stable/main")
        );
        assert!(cmdline.contains(
            "modloop=https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot/modloop-virt"
        ));
    }

    #[test]
    fn fedora_repo_release_matches_the_cached_kernel_release() {
        let source = Distro::Fedora.source();
        let kernel_release = release_in(source.kernel.url);
        assert_eq!(kernel_release, release_in(source.initramfs.url));
        let cmdline = Distro::Fedora.cmdline();
        let repo = cmdline
            .split_whitespace()
            .find_map(|arg| arg.strip_prefix("inst.repo="))
            .expect("inst.repo");
        assert_eq!(release_in(repo), kernel_release);
    }

    #[test]
    fn fedora_files_match_the_pxeboot_layout() {
        let source = Distro::Fedora.source();
        assert_eq!(source.kernel.file_name, "vmlinuz");
        assert_eq!(source.initramfs.file_name, "initrd.img");
        assert!(source.kernel.url.contains("/images/pxeboot/"));
    }

    #[test]
    fn network_installers_configure_the_nic_with_dhcp_in_the_initramfs() {
        assert!(
            Distro::Alpine
                .cmdline()
                .split_whitespace()
                .any(|a| a == "ip=dhcp")
        );
        assert!(
            Distro::Fedora
                .cmdline()
                .split_whitespace()
                .any(|a| a == "ip=dhcp")
        );
        assert!(Distro::Fedora.cmdline().contains("rd.neednet=1"));
        assert!(Distro::Fedora.cmdline().contains("inst.text"));
    }

    #[test]
    fn debian_needs_no_extra_kernel_arguments() {
        assert_eq!(Distro::Debian.cmdline(), "");
    }

    #[test]
    fn cmdlines_have_no_stray_whitespace() {
        for distro in ALL_DISTROS {
            let cmdline = distro.cmdline();
            assert_eq!(cmdline, cmdline.trim());
            assert!(!cmdline.contains("  "));
        }
    }

    #[test]
    fn needs_download_when_missing_empty_or_updating() {
        let dir = TempDir::new("quick-needs");
        let missing = dir.path().join("missing");
        let empty = dir.path().join("empty");
        let full = dir.path().join("full");
        fs::write(&empty, b"").unwrap();
        fs::write(&full, b"x").unwrap();
        assert!(needs_download(&missing, false));
        assert!(needs_download(&empty, false));
        assert!(!needs_download(&full, false));
        assert!(needs_download(&full, true));
        assert!(needs_download(&missing, true));
    }

    /// Records every requested URL and writes the URL itself as the file contents.
    struct FakeFetcher {
        urls: RefCell<Vec<String>>,
        fail_on: Option<&'static str>,
    }

    impl FakeFetcher {
        fn new() -> Self {
            FakeFetcher {
                urls: RefCell::new(Vec::new()),
                fail_on: None,
            }
        }

        fn fetch(&self, url: &str, dest: &Path) -> Result<(), String> {
            self.urls.borrow_mut().push(url.to_string());
            if self.fail_on.is_some_and(|needle| url.contains(needle)) {
                return Err(format!("download of {url} failed: injected"));
            }
            fs::write(dest, url).map_err(|e| e.to_string())
        }

        fn urls(&self) -> Vec<String> {
            self.urls.borrow().clone()
        }
    }

    #[test]
    fn first_ensure_fetches_kernel_then_initramfs_into_the_artifact_dir() {
        let dir = TempDir::new("quick-ensure");
        let artifacts_dir = dir.path().join("a").join("b");
        let fetcher = FakeFetcher::new();
        let artifacts = ensure_in(&artifacts_dir, Distro::Debian, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        let source = Distro::Debian.source();
        assert_eq!(
            fetcher.urls(),
            vec![source.kernel.url, source.initramfs.url]
        );
        assert_eq!(artifacts.kernel, artifacts_dir.join("linux"));
        assert_eq!(artifacts.initramfs, artifacts_dir.join("initrd.gz"));
        assert_eq!(
            fs::read_to_string(&artifacts.kernel).unwrap(),
            source.kernel.url
        );
    }

    #[test]
    fn cached_artifacts_are_not_fetched_again() {
        let dir = TempDir::new("quick-cached");
        let fetcher = FakeFetcher::new();
        ensure_in(dir.path(), Distro::Alpine, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        ensure_in(dir.path(), Distro::Alpine, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        assert_eq!(fetcher.urls().len(), 2);
    }

    #[test]
    fn update_refetches_both_artifacts() {
        let dir = TempDir::new("quick-update");
        let fetcher = FakeFetcher::new();
        ensure_in(dir.path(), Distro::Fedora, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        ensure_in(dir.path(), Distro::Fedora, true, |u, d| fetcher.fetch(u, d)).unwrap();
        assert_eq!(fetcher.urls().len(), 4);
    }

    #[test]
    fn a_zero_length_cached_file_is_fetched_again() {
        let dir = TempDir::new("quick-zero");
        let fetcher = FakeFetcher::new();
        fs::write(dir.path().join("vmlinuz-virt"), b"kernel").unwrap();
        fs::write(dir.path().join("initramfs-virt"), b"").unwrap();
        ensure_in(dir.path(), Distro::Alpine, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        assert_eq!(fetcher.urls(), vec![Distro::Alpine.source().initramfs.url]);
        assert_eq!(
            fs::read(dir.path().join("vmlinuz-virt")).unwrap(),
            b"kernel"
        );
    }

    #[test]
    fn a_failed_kernel_download_stops_before_the_initramfs() {
        let dir = TempDir::new("quick-fail");
        let mut fetcher = FakeFetcher::new();
        fetcher.fail_on = Some("vmlinuz");
        let err = ensure_in(dir.path(), Distro::Fedora, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .err()
        .unwrap();
        assert!(err.contains("injected"), "{err}");
        assert_eq!(fetcher.urls().len(), 1);
    }

    #[test]
    fn a_failed_initramfs_download_keeps_the_kernel_for_next_time() {
        let dir = TempDir::new("quick-fail2");
        let mut fetcher = FakeFetcher::new();
        fetcher.fail_on = Some("initrd.gz");
        assert!(
            ensure_in(dir.path(), Distro::Debian, false, |u, d| fetcher
                .fetch(u, d))
            .is_err()
        );
        assert!(dir.path().join("linux").exists());
        assert!(!dir.path().join("initrd.gz").exists());
        let retry = FakeFetcher::new();
        ensure_in(dir.path(), Distro::Debian, false, |u, d| retry.fetch(u, d)).unwrap();
        assert_eq!(retry.urls(), vec![Distro::Debian.source().initramfs.url]);
    }

    #[test]
    fn ensure_fails_cleanly_when_the_cache_directory_cannot_be_created() {
        let dir = TempDir::new("quick-nodir");
        let blocker = dir.path().join("file");
        fs::write(&blocker, b"x").unwrap();
        let fetcher = FakeFetcher::new();
        let err = ensure_in(&blocker.join("sub"), Distro::Debian, false, |u, d| {
            fetcher.fetch(u, d)
        })
        .err()
        .unwrap();
        assert!(err.starts_with("create "), "{err}");
        assert!(fetcher.urls().is_empty());
    }

    #[test]
    fn part_file_sits_next_to_the_destination_with_a_part_extension() {
        assert_eq!(
            part_path(Path::new("/c/linux")),
            PathBuf::from("/c/linux.part")
        );
        assert_eq!(
            part_path(Path::new("/c/initrd.gz")),
            PathBuf::from("/c/initrd.part")
        );
        assert_eq!(
            part_path(Path::new("/c/x")).parent(),
            Path::new("/c/x").parent()
        );
    }

    #[test]
    fn atomic_write_copies_everything_and_leaves_no_part_file() {
        let dir = TempDir::new("quick-atomic");
        let dest = dir.path().join("linux");
        let body: Vec<u8> = (0..(3 * DOWNLOAD_CHUNK_LEN + 7))
            .map(|i| (i % 251) as u8)
            .collect();
        write_atomically(Cursor::new(body.clone()), &dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
        assert!(!part_path(&dest).exists());
    }

    #[test]
    fn atomic_write_of_an_empty_body_creates_an_empty_file() {
        let dir = TempDir::new("quick-atomic-empty");
        let dest = dir.path().join("empty");
        write_atomically(Cursor::new(Vec::new()), &dest).unwrap();
        assert_eq!(fs::metadata(&dest).unwrap().len(), 0);
        assert!(
            needs_download(&dest, false),
            "an empty result is treated as missing"
        );
    }

    #[test]
    fn atomic_write_replaces_an_existing_file() {
        let dir = TempDir::new("quick-atomic-replace");
        let dest = dir.path().join("linux");
        fs::write(&dest, b"old contents that are longer").unwrap();
        write_atomically(Cursor::new(b"new".to_vec()), &dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"new");
    }

    struct BreaksAfter {
        remaining: Vec<u8>,
    }

    impl Read for BreaksAfter {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.remaining.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionReset,
                    "connection reset",
                ));
            }
            let n = self.remaining.len().min(buf.len());
            buf[..n].copy_from_slice(&self.remaining[..n]);
            self.remaining.drain(..n);
            Ok(n)
        }
    }

    #[test]
    fn interrupted_download_leaves_neither_part_file_nor_destination() {
        let dir = TempDir::new("quick-atomic-fail");
        let dest = dir.path().join("linux");
        let err = write_atomically(
            BreaksAfter {
                remaining: vec![7; 1000],
            },
            &dest,
        )
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionReset);
        assert!(!part_path(&dest).exists());
        assert!(!dest.exists(), "ensure must not find a truncated artifact");
    }

    #[test]
    fn interrupted_download_keeps_the_previous_good_file() {
        let dir = TempDir::new("quick-atomic-keep");
        let dest = dir.path().join("linux");
        fs::write(&dest, b"good").unwrap();
        assert!(
            write_atomically(
                BreaksAfter {
                    remaining: vec![7; 10]
                },
                &dest
            )
            .is_err()
        );
        assert_eq!(fs::read(&dest).unwrap(), b"good");
        assert!(!part_path(&dest).exists());
    }

    #[test]
    fn atomic_write_into_a_missing_directory_fails() {
        let dir = TempDir::new("quick-atomic-nodir");
        let dest = dir.path().join("missing").join("linux");
        assert!(write_atomically(Cursor::new(b"x".to_vec()), &dest).is_err());
    }
}
