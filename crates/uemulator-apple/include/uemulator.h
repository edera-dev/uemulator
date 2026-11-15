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

typedef struct {
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
// Stops the guest, waits for its threads and frees the machine. No callback runs after it returns.
void ua_destroy(ua_machine *machine);

#endif
