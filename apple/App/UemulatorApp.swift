import SwiftUI
import WebKit

@main
struct UemulatorApp: App {
    var body: some Scene {
        WindowGroup {
            MachineView()
        }
    }
}

// M2's one machine: the Alpine demo (the web page's demo machine, bundled by
// hack/apple/bundle-app.sh), booted through the bundled UEFI firmware on either engine.
@MainActor
final class MachineModel: ObservableObject {
    @Published var state: EngineState = .stopped("")
    @Published var startedAt: Date?
    @Published private(set) var kind: EngineKind

    let terminal = TerminalSession()
    private let native = NativeEngine()
    private var webKitEngine: WebKitEngine?
    // UserDefaults also reads `-engine native|webkit` from the command line.
    private static let kindKey = "engine"

    init() {
        kind = EngineKind(rawValue: UserDefaults.standard.string(forKey: Self.kindKey) ?? "") ?? .platformDefault
        wire(native)
        terminal.onInput = { [weak self] bytes in self?.engine.send(bytes) }
        terminal.onResize = { [weak self] cols, rows in self?.engine.resize(cols: cols, rows: rows) }
        if ProcessInfo.processInfo.arguments.contains("-autostart") { start() }
    }

    // Made on first use: it starts two loopback servers and a web view.
    var webKit: WebKitEngine {
        if let webKitEngine { return webKitEngine }
        let engine = WebKitEngine()
        wire(engine)
        webKitEngine = engine
        return engine
    }

    private var engine: any MachineEngine {
        switch kind {
        case .native: native
        case .webKit: webKit
        }
    }

    // Only the selected engine's callbacks reach the UI.
    private func wire(_ engine: any MachineEngine) {
        engine.onOutput = { [weak self, weak engine] bytes in
            guard let self, let engine, engine === self.engine else { return }
            self.terminal.feed(bytes)
        }
        engine.onState = { [weak self, weak engine] state in
            guard let self, let engine, engine === self.engine else { return }
            self.state = state
            if case .running = state { self.startedAt = Date() }
        }
    }

    var isRunning: Bool {
        switch state {
        case .starting, .running: true
        case .stopped: false
        }
    }

    func select(_ newKind: EngineKind) {
        guard newKind != kind else { return }
        engine.stop()
        kind = newKind
        UserDefaults.standard.set(kind.rawValue, forKey: Self.kindKey)
        state = .stopped("")
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
        engine.start(MachineSettings(
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
                // The WebKit engine's web view: in the hierarchy but invisible, never the UI.
                .background {
                    if model.kind == .webKit {
                        HiddenWebView(webView: model.webKit.webView).frame(width: 1, height: 1).opacity(0)
                    }
                }
                .navigationTitle("Alpine Linux")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    // Toolbars show a Label's icon alone by default; the chip is mostly its text.
                    ToolbarItem { StateChip(state: model.state, startedAt: model.startedAt).labelStyle(.titleAndIcon) }
                    ToolbarItem {
                        Menu {
                            Picker("Engine", selection: Binding(get: { model.kind }, set: { model.select($0) })) {
                                ForEach(EngineKind.allCases) { Text($0.title).tag($0) }
                            }
                        } label: {
                            Label("Engine", systemImage: "cpu")
                        }
                    }
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

// The web page's state chip: the uptime and how the machine runs, or why it stopped.
struct StateChip: View {
    let state: EngineState
    let startedAt: Date?

    var body: some View {
        switch state {
        case .starting:
            Label("Starting", systemImage: "hourglass")
        case .running(let detail):
            TimelineView(.periodic(from: .now, by: 1)) { context in
                let seconds = Int(context.date.timeIntervalSince(startedAt ?? context.date))
                // The green dot says running; the text stays short enough for a phone's toolbar.
                Label("\(seconds / 60):\(String(format: "%02d", seconds % 60)) · \(detail)", systemImage: "circle.fill")
                    .foregroundStyle(.green)
            }
        case .stopped(let reason):
            Label(reason.isEmpty ? "Stopped" : reason, systemImage: "stop.circle")
                .foregroundStyle(.secondary)
        }
    }
}

#if os(macOS)
struct HiddenWebView: NSViewRepresentable {
    let webView: WKWebView
    func makeNSView(context: Context) -> WKWebView { webView }
    func updateNSView(_ view: WKWebView, context: Context) {}
}
#else
struct HiddenWebView: UIViewRepresentable {
    let webView: WKWebView
    func makeUIView(context: Context) -> WKWebView { webView }
    func updateUIView(_ view: WKWebView, context: Context) {}
}
#endif
