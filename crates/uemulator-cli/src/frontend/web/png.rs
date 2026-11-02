//! A minimal PNG writer (ISO/IEC 15948): 8-bit RGB (screens) or RGBA (pointer images), one
//! IDAT, each row filtered with `Sub`.
//! Browsers decode PNG natively and off the main thread (`createImageBitmap`), and a console's
//! flat colours deflate to a few percent, so a screen update is small even off loopback.

use std::io::Write;

use flate2::Compression;
use flate2::write::ZlibEncoder;
use uemulator::{BYTES_PER_PIXEL, Rect};

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
const BIT_DEPTH: u8 = 8;
const COLOR_TYPE_RGB: u8 = 2;
const COLOR_TYPE_RGBA: u8 = 6;
const COMPRESSION_DEFLATE: u8 = 0;
const FILTER_METHOD_ADAPTIVE: u8 = 0;
const INTERLACE_NONE: u8 = 0;
/// Row filter `Sub`: each byte minus the same channel of the pixel to its left. Runs of one
/// colour become runs of zeros, which deflate well, for the cost of one subtraction per byte.
const FILTER_SUB: u8 = 1;
const RGB_BYTES: usize = 3;
const RGBA_BYTES: usize = 4;

/// Encodes `width` x `height` pixels in the library's frame format (blue, green, red, unused;
/// rows packed) as a PNG.
pub fn encode_bgrx(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    let row_bytes = width as usize * BYTES_PER_PIXEL;
    encode(width, height, pixels.chunks_exact(row_bytes), false)
}

/// Encodes the `rect` of a screen whose rows are `stride` bytes apart (a `ScreenMirror`) as a
/// PNG, without copying it out first.
pub fn encode_bgrx_rect(pixels: &[u8], stride: usize, rect: Rect) -> Vec<u8> {
    let left = rect.x as usize * BYTES_PER_PIXEL;
    let row_bytes = rect.width as usize * BYTES_PER_PIXEL;
    let rows = (rect.y..rect.y + rect.height).map(|row| {
        let start = row as usize * stride + left;
        &pixels[start..start + row_bytes]
    });
    encode(rect.width, rect.height, rows, false)
}

/// Encodes `width` x `height` pixels of blue, green, red, alpha (a `CursorImage`) as a PNG with
/// alpha.
pub fn encode_bgra(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    let row_bytes = width as usize * BYTES_PER_PIXEL;
    encode(width, height, pixels.chunks_exact(row_bytes), true)
}

fn encode<'a>(
    width: u32,
    height: u32,
    rows: impl Iterator<Item = &'a [u8]>,
    alpha: bool,
) -> Vec<u8> {
    let channels = if alpha { RGBA_BYTES } else { RGB_BYTES };
    let row_pixels = width as usize;
    let mut filtered = Vec::with_capacity((1 + row_pixels * channels) * height as usize);
    for row in rows.take(height as usize) {
        filtered.push(FILTER_SUB);
        let mut left = [0u8; RGBA_BYTES];
        for pixel in row.as_chunks::<BYTES_PER_PIXEL>().0 {
            let rgba = [pixel[2], pixel[1], pixel[0], pixel[3]];
            for channel in 0..channels {
                filtered.push(rgba[channel].wrapping_sub(left[channel]));
            }
            left = rgba;
        }
    }
    // Fast compression: the encoder sits between the guest drawing and the user seeing it.
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder
        .write_all(&filtered)
        .expect("writing to a Vec cannot fail");
    let compressed = encoder.finish().expect("writing to a Vec cannot fail");

    let mut png = SIGNATURE.to_vec();
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[
        BIT_DEPTH,
        if alpha {
            COLOR_TYPE_RGBA
        } else {
            COLOR_TYPE_RGB
        },
        COMPRESSION_DEFLATE,
        FILTER_METHOD_ADAPTIVE,
        INTERLACE_NONE,
    ]);
    chunk(&mut png, b"IHDR", &header);
    chunk(&mut png, b"IDAT", &compressed);
    chunk(&mut png, b"IEND", &[]);
    png
}

/// Appends a chunk: length, type, data, CRC-32 of type and data.
fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    let mut crc = flate2::Crc::new();
    crc.update(kind);
    crc.update(data);
    png.extend_from_slice(&crc.sum().to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::ZlibDecoder;
    use std::io::Read;

    /// (type, data) of every chunk, checking each CRC on the way.
    fn chunks(png: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
        assert_eq!(png[..8], SIGNATURE);
        let mut out = Vec::new();
        let mut at = 8;
        while at < png.len() {
            let length = u32::from_be_bytes(png[at..at + 4].try_into().unwrap()) as usize;
            let kind: [u8; 4] = png[at + 4..at + 8].try_into().unwrap();
            let data = png[at + 8..at + 8 + length].to_vec();
            let stored =
                u32::from_be_bytes(png[at + 8 + length..at + 12 + length].try_into().unwrap());
            let mut crc = flate2::Crc::new();
            crc.update(&kind);
            crc.update(&data);
            assert_eq!(crc.sum(), stored, "{:?}", std::str::from_utf8(&kind));
            out.push((kind, data));
            at += 12 + length;
        }
        out
    }

    #[test]
    fn the_image_round_trips_through_the_sub_filter() {
        let (width, height) = (3u32, 2u32);
        let pixels: Vec<u8> = (0..width * height)
            .flat_map(|i| [i as u8 * 10, i as u8 * 20 + 1, i as u8 * 30 + 2, 0xff])
            .collect();
        let png = encode_bgrx(width, height, &pixels);
        let chunks = chunks(&png);
        let kinds: Vec<_> = chunks.iter().map(|(kind, _)| *kind).collect();
        assert_eq!(kinds, [*b"IHDR", *b"IDAT", *b"IEND"]);
        let header = &chunks[0].1;
        assert_eq!(u32::from_be_bytes(header[0..4].try_into().unwrap()), width);
        assert_eq!(u32::from_be_bytes(header[4..8].try_into().unwrap()), height);
        assert_eq!(header[8..], [8, 2, 0, 0, 0]);
        let mut filtered = Vec::new();
        ZlibDecoder::new(&chunks[1].1[..])
            .read_to_end(&mut filtered)
            .unwrap();
        let row_bytes = 1 + width as usize * 3;
        assert_eq!(filtered.len(), row_bytes * height as usize);
        for (y, row) in filtered.chunks(row_bytes).enumerate() {
            assert_eq!(row[0], FILTER_SUB);
            let mut left = [0u8; 3];
            for x in 0..width as usize {
                let mut rgb = [0u8; 3];
                for c in 0..3 {
                    rgb[c] = row[1 + x * 3 + c].wrapping_add(left[c]);
                }
                let source = &pixels[(y * width as usize + x) * 4..][..4];
                assert_eq!(rgb, [source[2], source[1], source[0]], "pixel {x},{y}");
                left = rgb;
            }
        }
    }

    #[test]
    fn a_rectangle_of_a_larger_screen_encodes_as_its_own_image() {
        let (width, height) = (4u32, 3u32);
        let pixels: Vec<u8> = (0..width * height)
            .flat_map(|i| [i as u8, i as u8 + 100, i as u8 + 200, 0])
            .collect();
        let rect = Rect {
            x: 1,
            y: 1,
            width: 2,
            height: 2,
        };
        let mut packed = Vec::new();
        for row in 1..3 {
            let start = (row * width as usize + 1) * BYTES_PER_PIXEL;
            packed.extend_from_slice(&pixels[start..start + 2 * BYTES_PER_PIXEL]);
        }
        assert_eq!(
            encode_bgrx_rect(&pixels, width as usize * BYTES_PER_PIXEL, rect),
            encode_bgrx(2, 2, &packed)
        );
    }

    #[test]
    fn a_pointer_image_keeps_its_alpha() {
        let pixels = [0x10, 0x20, 0x30, 0x80, 0x10, 0x20, 0x30, 0x00];
        let png = encode_bgra(2, 1, &pixels);
        let chunks = chunks(&png);
        assert_eq!(chunks[0].1[8..], [8, 6, 0, 0, 0]);
        let mut filtered = Vec::new();
        ZlibDecoder::new(&chunks[1].1[..])
            .read_to_end(&mut filtered)
            .unwrap();
        // Sub filter: the second pixel is stored as its difference from the first.
        assert_eq!(
            filtered,
            [FILTER_SUB, 0x30, 0x20, 0x10, 0x80, 0, 0, 0, 0x80]
        );
    }

    #[test]
    fn a_flat_screen_compresses_to_almost_nothing() {
        let (width, height) = (1280u32, 800u32);
        let pixels = vec![0x40; (width * height) as usize * 4];
        let png = encode_bgrx(width, height, &pixels);
        assert!(png.len() < 64 * 1024, "{} bytes", png.len());
    }
}
