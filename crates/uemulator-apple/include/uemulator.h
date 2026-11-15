// The C ABI of crates/uemulator-apple, the native engine of the Apple app
// (docs/plans/28-apple.md). Hand-written: keep it in step with src/lib.rs, whose constants and
// struct layout it mirrors. Not thread-safe per machine except where a function says so.
#ifndef UEMULATOR_H
#define UEMULATOR_H

#include <stddef.h>
#include <stdint.h>

// `ua_config.options` bits. The same meaning as the web build's OPTION_* (www/protocol.js), so the
// app passes one set of choices to either engine.
#define UA_OPTION_JIT (1u << 0)
#define UA_OPTION_ACPI (1u << 1)
#define UA_OPTION_RNG (1u << 3)
#define UA_OPTION_SERIAL_CONSOLE (1u << 4)
// A screen (virtio-gpu and the firmware's framebuffer), shown through ua_screen_update.
#define UA_OPTION_DISPLAY (1u << 8)
// A virtio keyboard and tablet, fed through ua_key and the pointer calls.
#define UA_OPTION_INPUT (1u << 9)
// With UA_OPTION_DISPLAY: the screen is the guest's console, not the terminal.
#define UA_OPTION_SCREEN_CONSOLE (1u << 10)
// A TPM 2.0; its seeds and NV live in the NVRAM store.
#define UA_OPTION_TPM (1u << 11)
// The firmware enforces Secure Boot, enrolling its keys into the NVRAM store on first use.
#define UA_OPTION_SECURE_BOOT (1u << 12)
// A virtio NIC behind the library's gateway (10.0.2.0/24, DHCP, NAT), carried over Wisp v1 to the
// app: start it with ua_network_start. The app's own bit, clear of the web build's OPTION_*.
#define UA_OPTION_NETWORK (1u << 16)

// ua_cursor_fn kinds: the guest has no pointer of its own (show the host's), hid it, or drew one.
#define UA_CURSOR_HOST 0
#define UA_CURSOR_HIDDEN 1
#define UA_CURSOR_IMAGE 2

// Results of ua_screen_update, ua_cursor_update and the input calls: done (or queued for the
// guest); nothing new (or the guest's driver is not listening yet); the machine has no such device.
#define UA_RESULT_DONE 1
#define UA_RESULT_NOTHING_NEW 0
#define UA_RESULT_NO_DEVICE (-1)

// `ua_exit_fn` reasons, the library's ExitReason.
#define UA_EXIT_SHUTDOWN 0
#define UA_EXIT_RESET 1
#define UA_EXIT_TRIPLE_FAULT 2
#define UA_EXIT_STOPPED 3
#define UA_EXIT_INTERNAL 4

typedef struct ua_machine ua_machine;

// Guest console output. Runs on the vCPU thread with the console locked: copy the bytes and
// return, never call back into the machine.
typedef void (*ua_output_fn)(void *context, const uint8_t *bytes, size_t len);
// The guest exited (once per ua_start). Runs on the machine's own thread; `message` is UTF-8,
// empty unless UA_EXIT_INTERNAL, and only valid during the call.
typedef void (*ua_exit_fn)(void *context, int32_t reason, const char *message);

// A rectangle of the screen in pixels, origin top left.
typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
} ua_rect;

// The whole screen as the app should now show it: `height` rows `stride` bytes apart, each pixel
// blue, green, red, unused. `rects` are the parts that differ from the previous call (all of it
// the first time and after a size change; none when the guest flushed unchanged pixels). Runs on
// the caller's thread inside ua_screen_update; the pointers are valid only during the call.
typedef void (*ua_screen_fn)(void *context, uint32_t width, uint32_t height, const uint8_t *pixels,
                             size_t stride, const ua_rect *rects, size_t rect_count);
// The guest's pointer. For UA_CURSOR_IMAGE, `pixels` holds `height` rows of `width` pixels, blue,
// green, red, alpha (straight, not premultiplied), and (`hot_x`, `hot_y`) is the pointer's tip.
// Runs inside ua_cursor_update; `pixels` is valid only during the call.
typedef void (*ua_cursor_fn)(void *context, int32_t kind, uint32_t width, uint32_t height,
                             uint32_t hot_x, uint32_t hot_y, const uint8_t *pixels, size_t len);

// An image file the guest gets as a virtio-blk disk: raw (an ISO too) or udisk, not an overlay.
typedef struct {
    // UTF-8, NUL-terminated. On macOS the app must hold the file's security scope until ua_destroy.
    const char *path;
    // Nonzero: the guest sees a read-only disk (what a CD should be).
    int32_t read_only;
} ua_disk;

typedef struct {
    // NULL with length 0: no kernel; the UEFI firmware boots the disks.
    const uint8_t *kernel;
    size_t kernel_len;
    // May be NULL with length 0.
    const uint8_t *initrd;
    size_t initrd_len;
    // The UEFI firmware image (uemulator-firmware.bin); NULL boots the kernel directly.
    const uint8_t *firmware;
    size_t firmware_len;
    // Appended to the library's default command line; NULL for none. UTF-8, NUL-terminated.
    const char *cmdline_append;
    // In order: vda, vdb, ... May be NULL with count 0.
    const ua_disk *disks;
    size_t disk_count;
    // The machine's nvram.kv (UEFI variables, TPM state), created if missing; NULL keeps them in
    // memory until the guest resets.
    const char *nvram_path;
    uint32_t memory_mib;
    uint32_t cpus;
    uint32_t options;
    uint16_t cols;
    uint16_t rows;
} ua_config;

// Builds a machine; it does not run until ua_start. The buffers are copied. On failure returns
// NULL and writes a NUL-terminated message into `error` (truncated to `error_capacity`).
ua_machine *ua_create(const ua_config *config, ua_output_fn output, ua_exit_fn exit, void *context,
                      char *error, size_t error_capacity);
// 1 if the machine runs generated code, 0 if it interprets: UA_OPTION_JIT asks for the JIT, and a
// host that cannot map executable memory (iOS) gets the interpreter instead.
int32_t ua_jit_active(const ua_machine *machine);
// Starts the guest on threads of its own and returns. 0, or -1 if it was already started.
int32_t ua_start(ua_machine *machine);
// Keystrokes for the guest's console. Any thread.
void ua_input(const ua_machine *machine, const uint8_t *bytes, size_t len);
// The terminal's size; reaches the guest on a virtio console (hvc0). Any thread.
void ua_resize(const ua_machine *machine, uint16_t cols, uint16_t rows);
// Asks the guest to stop; the exit callback follows with UA_EXIT_STOPPED. Any thread.
void ua_stop(const ua_machine *machine);
// 1 if the machine has a screen (UA_OPTION_DISPLAY).
int32_t ua_has_display(const ua_machine *machine);
// Brings the app's copy of the screen up to date, calling `screen` when anything changed. Poll it
// once per display refresh from the thread that draws; cheap when nothing changed.
int32_t ua_screen_update(const ua_machine *machine, ua_screen_fn screen, void *context);
// Calls `cursor` when the pointer image changed since the last call (and on the first).
int32_t ua_cursor_update(const ua_machine *machine, ua_cursor_fn cursor, void *context);
// A key by USB HID usage (keyboard page 0x07: 0x04 is A, 0xE0 left control). Any thread.
int32_t ua_key(const ua_machine *machine, uint16_t usage, int32_t pressed);
// The pointer at (`x`, `y`) in screen pixels (clamped to the screen). Any thread.
int32_t ua_pointer_move(const ua_machine *machine, uint32_t x, uint32_t y);
// `button` 0 left, 1 right, 2 middle, 3 back, 4 forward. Any thread.
int32_t ua_pointer_button(const ua_machine *machine, uint32_t button, int32_t pressed);
// The wheel, `clicks` notches, positive away from the user. Any thread.
int32_t ua_scroll(const ua_machine *machine, int32_t clicks);
// Releases every key and button the guest thinks is down (the view lost the focus). Any thread.
int32_t ua_release_all(const ua_machine *machine);
// Asks the guest for a `width` x `height` screen, as a monitor hotplug would; the guest decides.
int32_t ua_request_size(const ua_machine *machine, uint32_t width, uint32_t height);
// machine.toml (the CLI's format) at `path` as a JSON object, an empty one when the file does not
// exist; free it with ua_string_free. NULL on failure, with the message in `error`.
char *ua_machine_load(const char *path, char *error, size_t error_capacity);
// Merges `changes`, a JSON object (null removes a key), into machine.toml at `path` and writes it
// back atomically; keys the app does not know are kept. 0, or -1 with the message in `error`.
int32_t ua_machine_save(const char *path, const char *changes, char *error, size_t error_capacity);
void ua_string_free(char *text);
// A new, empty, sparse udisk image of `size_mib` at `path`; never replaces a file. 0, or -1.
int32_t ua_disk_create(const char *path, uint64_t size_mib, char *error, size_t error_capacity);
// One Wisp v1 packet from the machine's client, for the app's Wisp server. Runs on the network's
// thread; `packet` is valid only during the call.
typedef void (*ua_wisp_send_fn)(void *context, const uint8_t *packet, size_t len);
// Starts the network of a machine built with UA_OPTION_NETWORK; `send` gets the client's packets.
// UA_RESULT_NO_DEVICE without a NIC, UA_RESULT_NOTHING_NEW when it already runs.
int32_t ua_network_start(const ua_machine *machine, ua_wisp_send_fn send, void *context);
// The server's side, in order: the connection opened, each message (none skipped or reordered),
// the connection lost. Any thread.
int32_t ua_wisp_connecting(const ua_machine *machine);
int32_t ua_wisp_message(const ua_machine *machine, const uint8_t *bytes, size_t len);
int32_t ua_wisp_lost(const ua_machine *machine);
// Stops the guest, waits for its threads and frees the machine. No callback runs after it returns.
void ua_destroy(ua_machine *machine);

#endif
