//! Disk images on the host: opening raw and udisk images (with their chain of bases, local or over
//! HTTP), creating them, and `uemulator disk` (`create`, `convert`, `info`). The format is
//! `docs/spec/uemulator-disk-format.md`; the library implements it.

use std::fs::{self, File, OpenOptions};
use std::hash::{BuildHasher, Hasher};
use std::io;
use std::path::{Path, PathBuf};

use uemulator::{
    BlockBackend, ImageFile, ImageStorage, RawImage, RemoteImage, Udisk, UdiskInfo, UdiskOptions,
    UdiskUsage,
};

pub const UDISK_EXTENSION: &str = "udisk";
const SECTOR_BYTES: u64 = 512;
const MIB: u64 = 1 << 20;
/// How much `convert` reads and writes at a time.
const COPY_CHUNK_BYTES: usize = MIB as usize;
/// A raw image is written sparsely: a run of this many zero bytes is left as a hole.
const RAW_HOLE_BYTES: usize = 64 * 1024;
/// Longest chain of bases `open_disk` follows; a longer one is almost certainly a loop.
const MAX_CHAIN_DEPTH: usize = 16;
/// How long an HTTP range request may take before the read fails (the guest then sees an I/O
/// error for that request).
const HTTP_TIMEOUT_SECONDS: u64 = 60;

/// Where an image lives: a local file or an HTTP(S) URL.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Location {
    File(PathBuf),
    Url(String),
}

impl Location {
    pub fn parse(text: &str) -> Location {
        match is_url(text) {
            true => Location::Url(text.to_string()),
            false => Location::File(text.into()),
        }
    }

    /// Resolves a udisk backing locator found in the image at `self`: relative paths are
    /// relative to the overlay's directory (or URL).
    fn resolve(&self, locator: &str) -> Location {
        if is_url(locator) {
            return Location::Url(locator.to_string());
        }
        match self {
            Location::File(path) => Location::File(match Path::new(locator).is_absolute() {
                true => locator.into(),
                false => path.parent().unwrap_or(Path::new("")).join(locator),
            }),
            Location::Url(url) => Location::Url(join_url(url, locator)),
        }
    }
}

impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Location::File(path) => write!(f, "{}", path.display()),
            Location::Url(url) => f.write_str(url),
        }
    }
}

fn is_url(text: &str) -> bool {
    text.starts_with("http://") || text.starts_with("https://")
}

/// `relative` resolved against the URL `base` the way a browser resolves a relative link: an
/// absolute path replaces the path, anything else replaces the last segment.
fn join_url(base: &str, relative: &str) -> String {
    let after_scheme = base.find("://").map_or(0, |at| at + 3);
    let path_start = base[after_scheme..]
        .find('/')
        .map_or(base.len(), |at| after_scheme + at);
    if relative.starts_with('/') {
        return format!("{}{relative}", &base[..path_start]);
    }
    match base[path_start..].rfind('/') {
        Some(at) => format!("{}{relative}", &base[..path_start + at + 1]),
        None => format!("{base}/{relative}"),
    }
}

/// 16 bytes that differ between images: an overlay identifies its base by them. `RandomState`
/// is seeded from the OS's random source; the time and process id are mixed in so that two
/// images made in one process differ too.
pub fn random_id() -> [u8; 16] {
    let mut id = [0u8; 16];
    for (index, half) in id.chunks_mut(8).enumerate() {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_usize(index);
        hasher.write_u32(std::process::id());
        hasher.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_nanos()),
        );
        half.copy_from_slice(&hasher.finish().to_le_bytes());
    }
    id
}

/// Read-only storage over HTTP range requests. The server must answer `Range` with 206: one that
/// sends the whole file every time would make each read a full download.
fn open_url(url: &str) -> Result<RemoteImage, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(HTTP_TIMEOUT_SECONDS)))
        .build()
        .into();
    let response = agent
        .get(url)
        .header("Range", "bytes=0-0")
        .call()
        .map_err(|e| format!("{url}: {e}"))?;
    if response.status() != 206 {
        return Err(format!(
            "{url}: the server answered a range request with {} instead of 206; it must support \
             range requests to stream a disk",
            response.status()
        ));
    }
    let len = response
        .headers()
        .get("content-range")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.rsplit('/').next())
        .and_then(|total| total.trim().parse::<u64>().ok())
        .ok_or_else(|| format!("{url}: no total length in the Content-Range header"))?;
    let url = url.to_string();
    Ok(RemoteImage::new(len, move |offset, buf: &mut [u8]| {
        let last = offset + buf.len() as u64 - 1;
        let response = agent
            .get(&url)
            .header("Range", format!("bytes={offset}-{last}"))
            .call()
            .map_err(io::Error::other)?;
        if response.status() != 206 {
            return Err(io::Error::other(format!(
                "{url}: range request answered with {}",
                response.status()
            )));
        }
        io::Read::read_exact(&mut response.into_body().into_reader(), buf)
    }))
}

fn open_storage(location: &Location, read_only: bool) -> Result<Box<dyn ImageStorage>, String> {
    match location {
        Location::File(path) => ImageFile::open(path, read_only)
            .map(|file| Box::new(file) as Box<dyn ImageStorage>)
            .map_err(|e| e.to_string()),
        Location::Url(_) if !read_only => Err(format!("{location} can only be opened read-only")),
        Location::Url(url) => open_url(url).map(|remote| Box::new(remote) as Box<dyn ImageStorage>),
    }
}

/// Opens a raw or udisk image, and for an overlay its base, and the base's base, read-only.
/// Errors about a base name it; the caller names `location`.
pub fn open_disk(location: &Location, read_only: bool) -> Result<Box<dyn BlockBackend>, String> {
    open_chain(location, read_only, 0)
}

fn open_chain(
    location: &Location,
    read_only: bool,
    depth: usize,
) -> Result<Box<dyn BlockBackend>, String> {
    if depth > MAX_CHAIN_DEPTH {
        return Err(format!(
            "more than {MAX_CHAIN_DEPTH} bases; is there a loop?"
        ));
    }
    let mut storage = open_storage(location, read_only)?;
    let info = Udisk::probe(&mut *storage).map_err(|e| e.to_string())?;
    let Some(info) = info else {
        return Ok(Box::new(RawImage::new(storage)));
    };
    let base = match &info.backing {
        None => None,
        Some(backing) if backing.locator.is_empty() => {
            return Err("the overlay does not say where its base is".into());
        }
        Some(backing) => {
            let base_location = location.resolve(&backing.locator);
            Some(
                open_chain(&base_location, true, depth + 1)
                    .map_err(|e| format!("base {base_location}: {e}"))?,
            )
        }
    };
    let disk = Udisk::open(storage, base).map_err(|e| e.to_string())?;
    Ok(Box::new(disk))
}

/// Creates a udisk image at `path` (which must not exist), optionally over a base.
pub fn create_udisk(
    path: &Path,
    size: Option<u64>,
    backing: Option<&str>,
    cluster_size: Option<u64>,
) -> Result<UdiskInfo, String> {
    // First, so that a base given relative to a directory that does not exist yet resolves.
    if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    let base = backing
        .map(|locator| {
            open_disk(&Location::File(path.to_path_buf()).resolve(locator), true)
                .map_err(|e| format!("base {locator}: {e}"))
        })
        .transpose()?;
    let size = match (size, &base) {
        (Some(size), _) => size,
        (None, Some(base)) => base.len().div_ceil(SECTOR_BYTES) * SECTOR_BYTES,
        (None, None) => return Err("a disk without a base needs a size".into()),
    };
    let mut options = UdiskOptions::new(size, random_id());
    if let Some(bytes) = cluster_size {
        options = options.cluster_size(bytes);
    }
    if let Some(locator) = backing {
        options = options.backing_locator(locator);
    }
    let file = ImageFile::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    match Udisk::create(file, &options, base) {
        Ok(disk) => Ok(disk.info().clone()),
        Err(e) => {
            let _ = fs::remove_file(path);
            Err(format!("create {}: {e}", path.display()))
        }
    }
}

/// The first `len` bytes of a disk image, through its format.
pub fn read_start(path: &Path, len: u64) -> Result<Vec<u8>, String> {
    let mut disk = open_disk(&Location::File(path.to_path_buf()), true)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut start = vec![0; len.min(disk.len()) as usize];
    disk.read_at(0, &mut start)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(start)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ImageFormat {
    Udisk,
    Raw,
}

impl std::fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ImageFormat::Udisk => "udisk",
            ImageFormat::Raw => "raw",
        })
    }
}

impl ImageFormat {
    /// What `convert` writes when not told: raw for `.img` and `.raw`, else udisk.
    pub fn for_path(path: &Path) -> ImageFormat {
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("img" | "raw") => ImageFormat::Raw,
            _ => ImageFormat::Udisk,
        }
    }
}

/// Copies the disk at `source` (whatever its format, through all its bases) into a new image at
/// `destination`. The result is self-contained: an overlay comes out flattened. Written to a
/// `.part` file first, so a failed conversion leaves nothing behind.
pub fn convert(
    source: &Location,
    destination: &Path,
    format: ImageFormat,
    cluster_size: Option<u64>,
) -> Result<u64, String> {
    if destination.exists() {
        return Err(format!("{} exists", destination.display()));
    }
    let mut input = open_disk(source, true).map_err(|e| format!("open {source}: {e}"))?;
    let part = destination.with_extension(match destination.extension() {
        Some(extension) => format!("{}.part", extension.to_string_lossy()),
        None => "part".into(),
    });
    let _ = fs::remove_file(&part);
    let result = match format {
        ImageFormat::Udisk => copy_to_udisk(&mut *input, &part, cluster_size),
        ImageFormat::Raw => copy_to_raw(&mut *input, &part),
    }
    .and_then(|()| fs::rename(&part, destination).map_err(|e| e.to_string()));
    if result.is_err() {
        let _ = fs::remove_file(&part);
    }
    result
        .map(|()| input.len())
        .map_err(|e| format!("convert {source} to {}: {e}", destination.display()))
}

/// Calls `body(offset, bytes)` for the whole of `input` in order, a chunk at a time.
fn copy_chunks(
    input: &mut dyn BlockBackend,
    mut body: impl FnMut(u64, &[u8]) -> io::Result<()>,
) -> Result<(), String> {
    let mut buffer = vec![0; COPY_CHUNK_BYTES];
    let mut offset = 0;
    while offset < input.len() {
        let count = (input.len() - offset).min(COPY_CHUNK_BYTES as u64) as usize;
        input
            .read_at(offset, &mut buffer[..count])
            .map_err(|e| format!("read at {offset}: {e}"))?;
        body(offset, &buffer[..count]).map_err(|e| format!("write at {offset}: {e}"))?;
        offset += count as u64;
    }
    Ok(())
}

/// The udisk writer skips all-zero sub-blocks itself, and appending in guest order lays the
/// clusters out front to back, which is what a streamed image wants.
fn copy_to_udisk(
    input: &mut dyn BlockBackend,
    path: &Path,
    cluster_size: Option<u64>,
) -> Result<(), String> {
    let size = input.len().div_ceil(SECTOR_BYTES) * SECTOR_BYTES;
    let mut options = UdiskOptions::new(size, random_id());
    if let Some(bytes) = cluster_size {
        options = options.cluster_size(bytes);
    }
    let file = ImageFile::create(path).map_err(|e| e.to_string())?;
    let mut output = Udisk::create(file, &options, None).map_err(|e| e.to_string())?;
    copy_chunks(input, |offset, bytes| output.write_at(offset, bytes))?;
    output.flush().map_err(|e| e.to_string())
}

fn copy_to_raw(input: &mut dyn BlockBackend, path: &Path) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.set_len(input.len()).map_err(|e| e.to_string())?;
    copy_chunks(input, |offset, bytes| {
        for (index, piece) in bytes.chunks(RAW_HOLE_BYTES).enumerate() {
            if piece.iter().any(|byte| *byte != 0) {
                write_at(&file, piece, offset + (index * RAW_HOLE_BYTES) as u64)?;
            }
        }
        Ok(())
    })?;
    file.sync_all().map_err(|e| e.to_string())
}

#[cfg(unix)]
fn write_at(file: &File, data: &[u8], offset: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, data, offset)
}

#[cfg(windows)]
fn write_at(file: &File, mut data: &[u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !data.is_empty() {
        let count = file.seek_write(data, offset)?;
        data = &data[count..];
        offset += count as u64;
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / MIB as f64)
}

/// What `disk info` prints.
pub fn describe(location: &Location) -> Result<String, String> {
    let mut storage = open_storage(location, true).map_err(|e| format!("open {location}: {e}"))?;
    let storage_len = storage.len();
    let Some(info) = Udisk::probe(&mut *storage).map_err(|e| format!("{location}: {e}"))? else {
        return Ok(format!(
            "{location}\n  format    raw\n  size      {storage_len} bytes ({})\n",
            mib(storage_len)
        ));
    };
    let mut out = format!(
        "{location}\n  format    udisk\n  size      {} bytes ({})\n  cluster   {} KiB\n  id        {}\n  file      {}\n",
        info.virtual_size,
        mib(info.virtual_size),
        info.cluster_size / 1024,
        hex(&info.image_id),
        mib(storage_len),
    );
    if let Some(backing) = &info.backing {
        out.push_str(&format!(
            "  base      {}\n  base id   {}\n  base size {} bytes\n",
            match backing.locator.is_empty() {
                true => "(not recorded)".into(),
                false => location.resolve(&backing.locator).to_string(),
            },
            backing.image_id.map_or("(raw image)".into(), |id| hex(&id)),
            backing.size
        ));
    }
    // Usage reads only this image's map, but `Udisk::open` checks the base first.
    let usage = || -> Result<UdiskUsage, String> {
        let base = match &info.backing {
            Some(backing) if backing.locator.is_empty() => {
                return Err("the base is not recorded".into());
            }
            Some(backing) => Some(open_disk(&location.resolve(&backing.locator), true)?),
            None => None,
        };
        let mut disk = Udisk::open(storage, base).map_err(|e| e.to_string())?;
        disk.usage().map_err(|e| e.to_string())
    };
    match usage() {
        Ok(usage) => out.push_str(&format!(
            "  data      {}\n  zero      {}\n  {}{}\n",
            mib(usage.data_bytes),
            mib(usage.zero_bytes),
            match info.backing {
                Some(_) => "from base ",
                None => "unwritten ",
            },
            mib(usage.unallocated_bytes),
        )),
        Err(e) => out.push_str(&format!("  usage     unavailable: {e}\n")),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;

    fn write(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).unwrap();
    }

    fn read_all(location: &Location) -> Vec<u8> {
        let mut disk = open_disk(location, true).unwrap();
        let mut bytes = vec![0; disk.len() as usize];
        disk.read_at(0, &mut bytes).unwrap();
        bytes
    }

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|index| (index % 253) as u8).collect()
    }

    #[test]
    fn urls_join_like_links() {
        assert_eq!(
            join_url("http://h/a/b.udisk", "c.udisk"),
            "http://h/a/c.udisk"
        );
        assert_eq!(
            join_url("http://h/a/b.udisk", "/x/c.udisk"),
            "http://h/x/c.udisk"
        );
        assert_eq!(
            join_url("https://h:8080", "c.udisk"),
            "https://h:8080/c.udisk"
        );
        assert_eq!(join_url("http://h/", "c"), "http://h/c");
    }

    #[test]
    fn locators_resolve_against_the_overlay() {
        let overlay = Location::File("/m/disks/top.udisk".into());
        assert_eq!(
            overlay.resolve("base.udisk"),
            Location::File("/m/disks/base.udisk".into())
        );
        assert_eq!(
            overlay.resolve("/abs.img"),
            Location::File("/abs.img".into())
        );
        assert_eq!(
            overlay.resolve("https://x/y.udisk"),
            Location::Url("https://x/y.udisk".into())
        );
        let remote = Location::Url("http://h/d/top.udisk".into());
        assert_eq!(
            remote.resolve("base.udisk"),
            Location::Url("http://h/d/base.udisk".into())
        );
    }

    #[test]
    fn ids_differ() {
        assert_ne!(random_id(), random_id());
    }

    #[test]
    fn raw_to_udisk_and_back_round_trips_and_stays_sparse() {
        let dir = TempDir::new("images-round-trip");
        let raw = dir.path().join("in.img");
        let mut contents = vec![0u8; 4 * MIB as usize + 512];
        contents[MIB as usize..MIB as usize + 70_000].copy_from_slice(&pattern(70_000));
        *contents.last_mut().unwrap() = 1;
        write(&raw, &contents);
        let udisk = dir.path().join("out.udisk");
        convert(
            &Location::File(raw.clone()),
            &udisk,
            ImageFormat::for_path(&udisk),
            None,
        )
        .unwrap();
        assert!(
            fs::metadata(&udisk).unwrap().len() < MIB,
            "only the written clusters are stored"
        );
        assert_eq!(read_all(&Location::File(udisk.clone())), contents);
        let back = dir.path().join("back.img");
        convert(
            &Location::File(udisk),
            &back,
            ImageFormat::for_path(&back),
            None,
        )
        .unwrap();
        assert_eq!(fs::read(&back).unwrap(), contents);
        assert!(!dir.path().join("back.img.part").exists());
    }

    #[test]
    fn convert_refuses_to_overwrite() {
        let dir = TempDir::new("images-exists");
        let raw = dir.path().join("in.img");
        write(&raw, &[1; 512]);
        let error =
            convert(&Location::File(raw.clone()), &raw, ImageFormat::Raw, None).unwrap_err();
        assert!(error.contains("exists"), "{error}");
    }

    #[test]
    fn an_overlay_finds_its_base_and_flattens() {
        let dir = TempDir::new("images-overlay");
        let base = dir.path().join("base.img");
        write(&base, &pattern(2 * MIB as usize));
        let overlay = dir.path().join("sub").join("top.udisk");
        let info = create_udisk(&overlay, None, Some("../base.img"), None).unwrap();
        assert_eq!(info.virtual_size, 2 * MIB);
        {
            let mut disk = open_disk(&Location::File(overlay.clone()), false).unwrap();
            disk.write_at(1000, b"written to the overlay").unwrap();
            disk.flush().unwrap();
        }
        assert_eq!(
            fs::read(&base).unwrap(),
            pattern(2 * MIB as usize),
            "the base is untouched"
        );
        let mut expected = pattern(2 * MIB as usize);
        expected[1000..1022].copy_from_slice(b"written to the overlay");
        assert_eq!(read_all(&Location::File(overlay.clone())), expected);
        let flat = dir.path().join("flat.udisk");
        convert(&Location::File(overlay), &flat, ImageFormat::Udisk, None).unwrap();
        let info = Udisk::probe(&mut ImageFile::open(&flat, true).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(info.backing, None);
        assert_eq!(read_all(&Location::File(flat)), expected);
    }

    #[test]
    fn a_replaced_base_is_refused() {
        let dir = TempDir::new("images-replaced");
        let base = dir.path().join("base.udisk");
        create_udisk(&base, Some(MIB), None, None).unwrap();
        let overlay = dir.path().join("top.udisk");
        create_udisk(&overlay, None, Some("base.udisk"), None).unwrap();
        fs::remove_file(&base).unwrap();
        create_udisk(&base, Some(MIB), None, None).unwrap();
        let error = open_disk(&Location::File(overlay), true).err().unwrap();
        assert!(
            error.contains("not the image the overlay was made over"),
            "{error}"
        );
    }

    #[test]
    fn a_missing_base_names_both_files() {
        let dir = TempDir::new("images-missing-base");
        let base = dir.path().join("base.img");
        write(&base, &[0; 4096]);
        let overlay = dir.path().join("top.udisk");
        create_udisk(&overlay, None, Some("base.img"), None).unwrap();
        fs::remove_file(&base).unwrap();
        let error = open_disk(&Location::File(overlay), true).err().unwrap();
        assert!(
            error.starts_with("base ") && error.contains("base.img"),
            "{error}"
        );
    }

    #[test]
    fn create_needs_a_size_or_a_base() {
        let dir = TempDir::new("images-create");
        let error = create_udisk(&dir.path().join("x.udisk"), None, None, None).unwrap_err();
        assert!(error.contains("size"), "{error}");
        assert!(!dir.path().join("x.udisk").exists());
        let error = create_udisk(&dir.path().join("y.udisk"), Some(1000), None, None).unwrap_err();
        assert!(error.contains("512"), "{error}");
        assert!(
            !dir.path().join("y.udisk").exists(),
            "a failed create leaves no file"
        );
    }

    #[test]
    fn describe_reports_both_formats() {
        let dir = TempDir::new("images-describe");
        let raw = dir.path().join("a.img");
        write(&raw, &[0; 4096]);
        assert!(
            describe(&Location::File(raw))
                .unwrap()
                .contains("format    raw")
        );
        let image = dir.path().join("b.udisk");
        create_udisk(&image, Some(MIB), None, None).unwrap();
        let text = describe(&Location::File(image)).unwrap();
        assert!(
            text.contains("format    udisk") && text.contains("unwritten 1.0 MiB"),
            "{text}"
        );
    }
}
