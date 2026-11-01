// Layout of the SharedArrayBuffer the page and the worker share: a small block of Int32 control
// words, then a byte ring for keystrokes (page writes HEAD, worker reads TAIL).
export const CTRL = {
  HEAD: 0,
  TAIL: 1,
  STOP: 2,
  RESIZE_GENERATION: 3,
  COLS: 4,
  ROWS: 5,
};
export const RING_OFFSET = 64;
export const RING_BYTES = 1 << 16;
export const CONTROL_BYTES = RING_OFFSET + RING_BYTES;
export const EXIT_NAMES = {
  0: "guest powered off",
  1: "guest reset",
  2: "triple fault",
  3: "stopped",
  4: "internal emulator error",
};
// `uw_set_firmware` kinds and `uw_create` option bits: the FIRMWARE_* and OPTION_* constants in
// src/lib.rs.
export const FIRMWARE = {
  NONE: 0,
  UEFI: 1,
  BIOS: 2,
  REAL_MODE: 3,
};
export const OPTION = {
  JIT: 1 << 0,
  ACPI: 1 << 1,
  IOMMU: 1 << 2,
  RNG: 1 << 3,
  SERIAL_CONSOLE: 1 << 4,
  REBOOT: 1 << 5,
  REPLACE_CMDLINE: 1 << 6,
  DEBUG: 1 << 7,
};
// The first bytes of a crates/uemulator-firmware image (docs/plans/19-uefi-firmware.md); anything else
// picked as a firmware image is taken for a real-mode one.
export const UEFI_FIRMWARE_MAGIC = "UEMUFW64";
