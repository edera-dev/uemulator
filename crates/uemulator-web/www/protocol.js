// Layout of the SharedArrayBuffer the page and the worker share: a small block of Int32 control
// words, then a byte ring for terminal keystrokes (page writes HEAD, worker reads TAIL), then a
// ring of screen input events (page writes EVENT_HEAD, worker reads EVENT_TAIL), each EVENT_WORDS
// Int32: kind (EVENT.*) and two arguments. The worker's idle wait sleeps on HEAD, so the page
// notifies HEAD after writing either ring.
export const CTRL = {
  HEAD: 0,
  TAIL: 1,
  STOP: 2,
  RESIZE_GENERATION: 3,
  COLS: 4,
  ROWS: 5,
  // 1 while the page can take a screen update: the worker clears it as it sends one, the page
  // sets it again once that one is on the canvas. One update in flight, never a backlog.
  FRAME_ACK: 6,
  EVENT_HEAD: 7,
  EVENT_TAIL: 8,
};
export const RING_OFFSET = 64;
export const RING_BYTES = 1 << 16;
export const EVENT_OFFSET = RING_OFFSET + RING_BYTES;
export const EVENT_WORDS = 3;
export const EVENT_RECORDS = 4096;
export const CONTROL_BYTES = EVENT_OFFSET + EVENT_RECORDS * EVENT_WORDS * 4;
// Screen input event kinds: the EVENT_* constants in src/lib.rs.
export const EVENT = {
  KEY: 1, // USB HID usage, pressed
  POINTER_MOVE: 2, // x, y in screen pixels
  POINTER_BUTTON: 3, // 0 left, 1 right, 2 middle, 3 back, 4 forward; pressed
  WHEEL: 4, // clicks, positive away from the user
  RELEASE_ALL: 5,
  REQUEST_SIZE: 6, // width, height
};
// `host_cursor` kinds: the CURSOR_* constants in src/lib.rs.
export const CURSOR = {
  HOST: 0,
  HIDDEN: 1,
  IMAGE: 2,
};
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
  DISPLAY: 1 << 8,
  INPUT: 1 << 9,
  SCREEN_CONSOLE: 1 << 10,
};
// The first bytes of a crates/uemulator-firmware image (docs/plans/19-uefi-firmware.md); anything else
// picked as a firmware image is taken for a real-mode one.
export const UEFI_FIRMWARE_MAGIC = "UEMUFW64";
