#!/usr/bin/env python3
"""Builds crates/uemulator-firmware/src/font8x16.bin, the glyphs the firmware's graphical text
console draws, from GNU Unifont's .hex file:

    curl -LO https://unifoundry.com/pub/unifont/unifont-16.0.04/font-builds/unifont-16.0.04.hex.gz
    python3 hack/uefi/make-font.py unifont-16.0.04.hex.gz crates/uemulator-firmware/src/font8x16.bin

Unifont's fonts are dual licensed, SIL OFL 1.1 or GPL 2+ with the font embedding exception;
either fits this GPL-2 project. Only 8x16 (single-width) glyphs are taken: printable ASCII,
Latin-1, and the arrows, box drawing, block elements and geometric shapes UEFI names in
`EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL` (BOXDRAW_*, BLOCKELEMENT_*, GEOMETRICSHAPE_*, ARROW_*).

Output: entries of 18 bytes, sorted by code point: the code point as u16 little endian, then 16
rows, top first, most significant bit leftmost.
"""

import gzip
import struct
import sys

RANGES = [
    (0x0020, 0x007E),  # ASCII
    (0x00A0, 0x00FF),  # Latin-1
    (0x2190, 0x2193),  # arrows
    (0x2500, 0x257F),  # box drawing
    (0x2580, 0x259F),  # block elements
    (0x25B2, 0x25B2),  # up triangle
    (0x25BA, 0x25BA),  # right triangle
    (0x25BC, 0x25BC),  # down triangle
    (0x25C4, 0x25C4),  # left triangle
    (0xFFFD, 0xFFFD),  # replacement character, drawn for anything missing
]
ROWS = 16
NARROW_HEX_DIGITS = ROWS * 2


def wanted(code):
    return any(low <= code <= high for low, high in RANGES)


def main(source, output):
    opener = gzip.open if source.endswith(".gz") else open
    glyphs = {}
    with opener(source, "rt") as lines:
        for line in lines:
            code, _, bitmap = line.strip().partition(":")
            code = int(code, 16)
            if wanted(code) and len(bitmap) == NARROW_HEX_DIGITS:
                glyphs[code] = bytes.fromhex(bitmap)
    with open(output, "wb") as out:
        for code in sorted(glyphs):
            out.write(struct.pack("<H", code) + glyphs[code])
    print(f"{len(glyphs)} glyphs", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
