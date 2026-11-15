import SwiftUI

@main
struct UemulatorApp: App {
    var body: some Scene {
        WindowGroup {
            MachineView()
        }
    }
}

// M1's one machine: the Alpine demo (the web page's demo machine, bundled by
// hack/apple/bundle-app.sh), booted through the bundled UEFI firmware on the native engine.
@MainActor
final class MachineModel: ObservableObject {
    @Published var state: NativeEngine.State = .stopped("")
    @Published var startedAt: Date?

    let engine = NativeEngine()
    let terminal = TerminalSession()

    init() {
        engine.onOutput = { [weak self] bytes in self?.terminal.feed(bytes) }
        engine.onState = { [weak self] state in
            guard let self else { return }
            self.state = state
            if case .running = state { self.startedAt = Date() }
        }
        terminal.onInput = { [weak self] bytes in self?.engine.send(bytes) }
        terminal.onResize = { [weak self] cols, rows in self?.engine.resize(cols: cols, rows: rows) }
        if ProcessInfo.processInfo.arguments.contains("-autostart") { start() }
    }

    var isRunning: Bool {
        switch state {
        case .starting, .running: true
        case .stopped: false
        }
    }

    func start() {
        let resources = Bundle.main.resourceURL!
        guard let kernel = try? Data(contentsOf: resources.appendingPathComponent("demo/vmlinuz-virt")),
              let initrd = try? Data(contentsOf: resources.appendingPathComponent("demo/initramfs.gz"))
        else {
            state = .stopped("The demo kernel is not in the app bundle (WEB_DEMO=1 hack/web.sh, then rebuild)")
            return
        }
        let firmware = try? Data(contentsOf: resources.appendingPathComponent("uemulator-firmware.bin"))
        terminal.reset()
        let size = terminal.size
        engine.start(NativeEngine.Settings(
            kernel: kernel, initrd: initrd, firmware: firmware,
            cols: UInt16(clamping: size.cols), rows: UInt16(clamping: size.rows)))
    }

    func stop() {
        engine.stop()
    }
}

struct MachineView: View {
    @StateObject private var model = MachineModel()

    var body: some View {
        NavigationStack {
            TerminalPane(session: model.terminal)
                .background(Color.black)
                .navigationTitle("Alpine Linux")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    // Toolbars show a Label's icon alone by default; the chip is mostly its text.
                    ToolbarItem { StateChip(state: model.state, startedAt: model.startedAt).labelStyle(.titleAndIcon) }
                    ToolbarItem {
                        if model.isRunning {
                            Button("Stop", systemImage: "stop.fill") { model.stop() }
                        } else {
                            Button("Start", systemImage: "play.fill") { model.start() }
                        }
                    }
                }
        }
    }
}

// The web page's state chip: Starting, Running and the uptime (and which engine), or why it
// stopped.
struct StateChip: View {
    let state: NativeEngine.State
    let startedAt: Date?

    var body: some View {
        switch state {
        case .starting:
            Label("Starting", systemImage: "hourglass")
        case .running(let jit):
            TimelineView(.periodic(from: .now, by: 1)) { context in
                let seconds = Int(context.date.timeIntervalSince(startedAt ?? context.date))
                // The green dot says running; the text stays short enough for a phone's toolbar.
                Label("\(seconds / 60):\(String(format: "%02d", seconds % 60)) · \(jit ? "JIT" : "interpreted")",
                      systemImage: "circle.fill")
                    .foregroundStyle(.green)
            }
        case .stopped(let reason):
            Label(reason.isEmpty ? "Stopped" : reason, systemImage: "stop.circle")
                .foregroundStyle(.secondary)
        }
    }
}
