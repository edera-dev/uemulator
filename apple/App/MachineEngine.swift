import Foundation

// What the UI needs from an engine (docs/plans/28-apple.md, "Why there are two engines"):
// NativeEngine runs the library in this process, WebKitEngine runs the web build in a hidden web
// view. Both give the console, the screen and the keyboard and tablet.
@MainActor
protocol MachineEngine: ScreenInput {
    // Console bytes from the guest, batched.
    var onOutput: (([UInt8]) -> Void)? { get set }
    var onState: ((EngineState) -> Void)? { get set }
    // Where the guest's screen goes, when the settings give it one. Set before `start`.
    var screen: ScreenSurface? { get set }

    func start(_ settings: MachineSettings)
    // Returns at once; the machine is gone from the UI's point of view.
    func stop()
    func send(_ bytes: [UInt8])
    func resize(cols: Int, rows: Int)
    // Asks the guest for this screen size, as a monitor hotplug would.
    func requestSize(width: Int, height: Int)
}

enum EngineState: Equatable {
    case starting
    // `detail` says how it runs: "JIT", "interp", "wasm" (the WebKit engine's JIT).
    case running(detail: String)
    case stopped(String)
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
    // than the native interpreter (M0); macOS JITs natively, several times faster than WebKit.
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
    // The UEFI firmware image; nil boots the kernel directly.
    var firmware: Data?
    var cmdlineAppend = ""
    var memoryMib: UInt32 = 256
    var cpus: UInt32 = 1
    var serialConsole = true
    var jit = true
    var display = DisplayChoice.terminal
    var cols: UInt16 = 80
    var rows: UInt16 = 25

    // `uw_create`'s and `ua_create`'s option bits (www/protocol.js OPTION, include/uemulator.h).
    var options: UInt32 {
        var options = Option.acpi | Option.rng
        if jit { options |= Option.jit }
        if serialConsole { options |= Option.serialConsole }
        if display.hasScreen { options |= Option.display | Option.input }
        if display == .screen { options |= Option.screenConsole }
        return options
    }

    private enum Option {
        static let jit: UInt32 = 1 << 0
        static let acpi: UInt32 = 1 << 1
        static let rng: UInt32 = 1 << 3
        static let serialConsole: UInt32 = 1 << 4
        static let display: UInt32 = 1 << 8
        static let input: UInt32 = 1 << 9
        static let screenConsole: UInt32 = 1 << 10
    }
}
