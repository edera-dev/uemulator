import Foundation

// What the UI needs from an engine (docs/plans/28-apple.md, "Why there are two engines"):
// NativeEngine runs the library in this process, WebKitEngine runs the web build in a hidden web
// view. Both give the console, the screen and the keyboard and tablet.
@MainActor
protocol MachineEngine: ScreenInput {
    // Console bytes from the guest, batched.
    var onOutput: (([UInt8]) -> Void)? { get set }
    var onState: ((EngineState) -> Void)? { get set }
    // Emulator diagnostics (the library's log hook, the web worker's failures).
    var onLog: ((String) -> Void)? { get set }
    // Where the guest's screen goes, when the settings give it one. Set before `start`.
    var screen: ScreenSurface? { get set }

    func start(_ settings: MachineSettings)
    // Returns at once; the machine is gone from the UI's point of view.
    func stop()
    // Stops the machine into `MachineSettings.suspendTo` (docs/plans/29-snapshots.md): the state
    // goes to `.preparing("Suspending")`, then `.stopped(EngineState.suspendedReason)` once the
    // snapshot is complete, or `.stopped` with why it failed.
    func suspend()
    func send(_ bytes: [UInt8])
    func resize(cols: Int, rows: Int)
    // Asks the guest for this screen size, as a monitor hotplug would.
    func requestSize(width: Int, height: Int)
}

enum EngineState: Equatable {
    // Before the engine: fetching what the machine boots.
    case preparing(String)
    case starting
    // `detail` says how it runs: "JIT", "interp", "wasm" (the WebKit engine's JIT).
    case running(detail: String)
    case stopped(String)

    // What both engines report when the guest resets. The library's own in-place restart is off
    // (`MachineSettings.options`): the app starts the machine again instead, which decides what
    // boots afresh, so an installer that reboots comes back on the disk it installed to.
    static let resetReason = "Reset"
    // What both engines report once a suspend's snapshot is complete.
    static let suspendedReason = "Suspended"
    static let suspending = EngineState.preparing("Suspending")
}

// What the machine starts in: the CLI's `--firmware`, and `ua_config.firmware_kind` /
// www/protocol.js FIRMWARE.
enum FirmwareKind: String, CaseIterable, Identifiable {
    case uefi
    case bios
    case none

    var id: String { rawValue }

    var title: String {
        switch self {
        case .uefi: "UEFI"
        case .bios: "BIOS"
        case .none: "None"
        }
    }

    var code: UInt32 {
        switch self {
        case .none: 0
        case .uefi: 1
        case .bios: 2
        }
    }
}

enum EngineKind: String, CaseIterable, Identifiable {
    case native
    case webKit = "webkit"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .native: "Native"
        case .webKit: "WebKit"
        }
    }

    // iOS cannot JIT natively, and the web build's JIT in WebKit boots about 6.5 times faster
    // than the native interpreter (docs/plans/28-apple.md, "M0 results"); macOS JITs natively,
    // several times faster than WebKit.
    static var platformDefault: EngineKind {
        #if os(iOS)
        .webKit
        #else
        .native
        #endif
    }
}

// The CLI's `--display serial|framebuffer|both`.
enum DisplayChoice: String, CaseIterable, Identifiable {
    // No screen: the terminal is the console.
    case terminal
    // A screen with keyboard and tablet, and the screen is the console (init's shell is there).
    case screen
    // A screen with keyboard and tablet; the terminal stays the console.
    case both

    var id: String { rawValue }

    var title: String {
        switch self {
        case .terminal: "Terminal only"
        case .screen: "Screen"
        case .both: "Screen and terminal"
        }
    }

    var hasScreen: Bool { self != .terminal }
}

// An image file the guest gets as a disk: raw (an ISO too) or udisk.
struct DiskImage {
    var url: URL
    // A CD should be: an installer that finds its medium writable may offer it as a target.
    var readOnly: Bool
}

struct MachineSettings {
    // Empty: no kernel, and the UEFI firmware boots the disks.
    var kernel: Data
    var initrd: Data
    var disks: [DiskImage] = []
    // The machine's nvram.kv: UEFI variables and TPM state. Nil keeps them in memory.
    var nvram: URL?
    // Where `suspend` writes the machine's snapshot, and one to resume from instead of booting
    // (deleted by the engine once the machine runs again: its disks then move on).
    var suspendTo: URL?
    var resumeFrom: URL?
    var firmware = FirmwareKind.uefi
    // The UEFI firmware image, for `.uefi`.
    var firmwareImage: Data?
    // Booted by the UEFI firmware with the kernel as its dom0.
    var hypervisor: Data?
    var hypervisorCmdline = ""
    var cmdlineAppend = ""
    // `cmdlineAppend` replaces the library's default command line.
    var replaceCmdline = false
    var memoryMib: UInt32 = 256
    var cpus: UInt32 = 1
    var serialConsole = true
    var jit = true
    var display = DisplayChoice.terminal
    var tpm = false
    var secureBoot = false
    var iommu = false
    var rng = true
    var smt = true
    // The vCPUs take turns on one thread (native only: the WebKit engine always does).
    var timeshare = false
    // Seconds since 1970 for the RTC; 0 follows the host.
    var unixTime: UInt64 = 0
    // A NIC behind the gateway (DHCP, NAT to the internet through the app's Wisp server).
    var network = false
    var cols: UInt16 = 80
    var rows: UInt16 = 25

    // `uw_create`'s and `ua_create`'s option bits (www/protocol.js OPTION, include/uemulator.h).
    var options: UInt32 {
        // No reboot bit: a guest reset ends the run and the app starts it again
        // (`EngineState.resetReason`).
        var options = Option.acpi
        if rng { options |= Option.rng }
        if iommu { options |= Option.iommu }
        if replaceCmdline { options |= Option.replaceCmdline }
        if !smt { options |= Option.noSmt }
        if jit { options |= Option.jit }
        if serialConsole { options |= Option.serialConsole }
        if display.hasScreen { options |= Option.display | Option.input }
        if display == .screen { options |= Option.screenConsole }
        if tpm { options |= Option.tpm }
        if secureBoot { options |= Option.secureBoot }
        // No network bit: the native engine adds UA_OPTION_NETWORK, an app-only bit, and the web
        // build sets its NIC up with uw_set_network from the boot message's `network`.
        return options
    }

    private enum Option {
        static let jit: UInt32 = 1 << 0
        static let acpi: UInt32 = 1 << 1
        static let iommu: UInt32 = 1 << 2
        static let rng: UInt32 = 1 << 3
        static let replaceCmdline: UInt32 = 1 << 6
        static let noSmt: UInt32 = 1 << 13
        static let serialConsole: UInt32 = 1 << 4
        static let display: UInt32 = 1 << 8
        static let input: UInt32 = 1 << 9
        static let screenConsole: UInt32 = 1 << 10
        static let tpm: UInt32 = 1 << 11
        static let secureBoot: UInt32 = 1 << 12
    }
}
