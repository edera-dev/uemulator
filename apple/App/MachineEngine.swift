import Foundation

// What the UI needs from an engine (docs/plans/28-apple.md, "Why there are two engines"):
// NativeEngine runs the library in this process, WebKitEngine runs the web build in a hidden web
// view. Both speak the console only so far.
@MainActor
protocol MachineEngine: AnyObject {
    // Console bytes from the guest, batched.
    var onOutput: (([UInt8]) -> Void)? { get set }
    var onState: ((EngineState) -> Void)? { get set }

    func start(_ settings: MachineSettings)
    // Returns at once; the machine is gone from the UI's point of view.
    func stop()
    func send(_ bytes: [UInt8])
    func resize(cols: Int, rows: Int)
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

struct MachineSettings {
    var kernel: Data
    var initrd: Data
    // The UEFI firmware image; nil boots the kernel directly.
    var firmware: Data?
    var cmdlineAppend = ""
    var memoryMib: UInt32 = 256
    var cpus: UInt32 = 1
    var serialConsole = true
    var jit = true
    var cols: UInt16 = 80
    var rows: UInt16 = 25
}
