import SwiftUI
import UniformTypeIdentifiers
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
final class MachineModel: ObservableObject, ScreenInput {
    enum View: String, CaseIterable, Identifiable {
        case terminal
        case screen
        var id: String { rawValue }
    }

    @Published var state: EngineState = .stopped("")
    @Published var startedAt: Date?
    @Published private(set) var kind: EngineKind
    @Published private(set) var display: DisplayChoice
    @Published var view: View = .terminal {
        // Keys held down while switching would never see their release.
        didSet { if view != oldValue { releaseAll() } }
    }

    let terminal = TerminalSession()
    let screen = ScreenSurface()
    private let native = NativeEngine()
    private var webKitEngine: WebKitEngine?
    // UserDefaults also reads `-engine native|webkit` from the command line.
    private static let kindKey = "engine"
    // Also `-display terminal|screen|both` on the command line.
    private static let displayKey = "display"

    // `-debug-log NAME` (in the app's Documents): the guest's console output, and every few
    // seconds a line saying what the screen is getting. For finding out, without the window, whether
    // a guest is stuck or its frames are not showing.
    private var debugLog: FileHandle?
    private var debugTimer: Timer?
    private static let debugInterval: TimeInterval = 5

    private func openDebugLog() {
        guard let name = UserDefaults.standard.string(forKey: "debug-log"), !name.isEmpty else { return }
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let url = documents.appendingPathComponent(name)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        debugLog = try? FileHandle(forWritingTo: url)
        let timer = Timer(timeInterval: Self.debugInterval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                let line = "\n[app] \(self.state) screen \(self.screen.width)x\(self.screen.height) frames \(self.screen.commits)\n"
                self.debugLog?.write(Data(line.utf8))
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        debugTimer = timer
    }

    init() {
        kind = EngineKind(rawValue: UserDefaults.standard.string(forKey: Self.kindKey) ?? "") ?? .platformDefault
        display = DisplayChoice(rawValue: UserDefaults.standard.string(forKey: Self.displayKey) ?? "") ?? .terminal
        view = display == .screen ? .screen : .terminal
        cdrom = Self.cdromArgument()
        wire(native)
        terminal.onInput = { [weak self] bytes in self?.engine.send(bytes) }
        terminal.onResize = { [weak self] cols, rows in self?.engine.resize(cols: cols, rows: rows) }
        openDebugLog()
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
        engine.screen = screen
        engine.onOutput = { [weak self, weak engine] bytes in
            guard let self, let engine, engine === self.engine else { return }
            self.terminal.feed(bytes)
            self.debugLog?.write(Data(bytes))
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

    func select(_ newDisplay: DisplayChoice) {
        guard newDisplay != display else { return }
        // The machine has its devices from the start: a change applies to the next one.
        engine.stop()
        display = newDisplay
        UserDefaults.standard.set(display.rawValue, forKey: Self.displayKey)
        view = display == .screen ? .screen : .terminal
        screen.reset()
        state = .stopped("")
    }

    // Asks the guest for a screen the size of the view, in pixels.
    func fitScreen(to size: CGSize, scale: CGFloat) {
        engine.requestSize(width: Int(size.width * scale), height: Int(size.height * scale))
    }

    // ScreenInput: whichever engine is selected.
    func key(usage: UInt16, pressed: Bool) { engine.key(usage: usage, pressed: pressed) }
    func pointerMove(x: Int, y: Int) { engine.pointerMove(x: x, y: y) }
    func pointerButton(_ button: Int, pressed: Bool) { engine.pointerButton(button, pressed: pressed) }
    func scroll(clicks: Int) { engine.scroll(clicks: clicks) }
    func releaseAll() { engine.releaseAll() }

    // A CD image to boot instead of the demo: picked (`bootISO`) or `-cdrom PATH` on the command
    // line (relative to the app's Documents, which a sandboxed app can read without asking).
    @Published private(set) var cdrom: URL?
    // A picked file is readable only inside its security scope, held while the machine may use it.
    private var scopedURL: URL?
    // Live ISOs want room: Fedora's 252 MB installer initramfs does not unpack in 1 GiB, and the
    // web page warns below 2 GiB for CDs (docs/plans/13-web.md).
    private static let cdromMemoryMib: UInt32 = 2048
    private static let demoMemoryMib: UInt32 = 256

    var title: String { cdrom?.lastPathComponent ?? "Alpine Linux" }

    func bootISO(_ url: URL) {
        engine.stop()
        scopedURL?.stopAccessingSecurityScopedResource()
        scopedURL = url.startAccessingSecurityScopedResource() ? url : nil
        cdrom = url
        start()
    }

    func bootDemo() {
        engine.stop()
        scopedURL?.stopAccessingSecurityScopedResource()
        scopedURL = nil
        cdrom = nil
        start()
    }

    private static func cdromArgument() -> URL? {
        guard let path = UserDefaults.standard.string(forKey: "cdrom"), !path.isEmpty else { return nil }
        if path.hasPrefix("/") { return URL(fileURLWithPath: path) }
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documents.appendingPathComponent(path)
    }

    func start() {
        let resources = Bundle.main.resourceURL!
        let firmware = try? Data(contentsOf: resources.appendingPathComponent("uemulator-firmware.bin"))
        let size = terminal.size
        if let cdrom {
            guard firmware != nil else {
                state = .stopped("The UEFI firmware is not in the app bundle; rebuild")
                return
            }
            terminal.reset()
            // No kernel: the firmware boots the CD's \EFI\BOOT\BOOTX64.EFI.
            engine.start(MachineSettings(
                kernel: Data(), initrd: Data(), disks: [DiskImage(url: cdrom, readOnly: true)], firmware: firmware,
                memoryMib: Self.cdromMemoryMib, display: display,
                cols: UInt16(clamping: size.cols), rows: UInt16(clamping: size.rows)))
            return
        }
        guard let kernel = try? Data(contentsOf: resources.appendingPathComponent("demo/vmlinuz-virt")),
              let initrd = try? Data(contentsOf: resources.appendingPathComponent("demo/initramfs.gz"))
        else {
            state = .stopped("The demo kernel is not in the app bundle (WEB_DEMO=1 hack/web.sh, then rebuild)")
            return
        }
        terminal.reset()
        engine.start(MachineSettings(
            kernel: kernel, initrd: initrd, firmware: firmware, memoryMib: Self.demoMemoryMib, display: display,
            cols: UInt16(clamping: size.cols), rows: UInt16(clamping: size.rows)))
    }

    func stop() {
        engine.stop()
    }
}

struct MachineView: View {
    // A phone's navigation bar has room for the title and two buttons: the state and the
    // Terminal | Screen switch go to the bottom bar there.
    #if os(iOS)
    private static let statusPlacement = ToolbarItemPlacement.bottomBar
    #else
    private static let statusPlacement = ToolbarItemPlacement.automatic
    #endif

    @StateObject private var model = MachineModel()
    @State private var viewSize = CGSize.zero
    @State private var choosingISO = false
    @State private var softwareKeyboard = false

    private var screenPane: ScreenPane {
        #if os(iOS)
        ScreenPane(surface: model.screen, input: model, showsSoftwareKeyboard: softwareKeyboard)
        #else
        ScreenPane(surface: model.screen, input: model)
        #endif
    }

    @Environment(\.displayScale) private var displayScale

    var body: some View {
        NavigationStack {
            GeometryReader { geometry in
                ZStack {
                    // Both stay alive so the terminal keeps its scrollback and the screen its image.
                    TerminalPane(session: model.terminal)
                        .opacity(model.view == .terminal ? 1 : 0)
                        .allowsHitTesting(model.view == .terminal)
                    if model.display.hasScreen {
                        screenPane
                            .opacity(model.view == .screen ? 1 : 0)
                            .allowsHitTesting(model.view == .screen)
                    }
                }
                .onAppear { viewSize = geometry.size }
                .onChange(of: geometry.size) { viewSize = geometry.size }
            }
                .background(Color.black)
                // The WebKit engine's web view: in the hierarchy but invisible, never the UI.
                .background {
                    if model.kind == .webKit {
                        HiddenWebView(webView: model.webKit.webView).frame(width: 1, height: 1).opacity(0)
                    }
                }
                .navigationTitle(model.title)
                .fileImporter(isPresented: $choosingISO, allowedContentTypes: [UTType(filenameExtension: "iso") ?? .data, .data]) { result in
                    if case .success(let url) = result { model.bootISO(url) }
                }
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    // Toolbars show a Label's icon alone by default; the chip is mostly its text.
                    ToolbarItem(placement: Self.statusPlacement) {
                        StateChip(state: model.state, startedAt: model.startedAt).labelStyle(.titleAndIcon)
                    }
                    #if os(iOS)
                    if model.display.hasScreen && model.view == .screen {
                        ToolbarItem(placement: Self.statusPlacement) {
                            Toggle(isOn: $softwareKeyboard) { Image(systemName: "keyboard") }
                                .toggleStyle(.button)
                        }
                    }
                    #endif
                    if model.display.hasScreen {
                        ToolbarItem(placement: Self.statusPlacement) {
                            Picker("View", selection: $model.view) {
                                Image(systemName: "terminal").tag(MachineModel.View.terminal)
                                Image(systemName: "display").tag(MachineModel.View.screen)
                            }
                            .pickerStyle(.segmented)
                        }
                    }
                    ToolbarItem {
                        Menu {
                            Picker("Engine", selection: Binding(get: { model.kind }, set: { model.select($0) })) {
                                ForEach(EngineKind.allCases) { Text($0.title).tag($0) }
                            }
                            Button("Boot ISO…", systemImage: "opticaldisc") { choosingISO = true }
                            Button("Boot Alpine Demo", systemImage: "play.rectangle") { model.bootDemo() }
                            Divider()
                            Picker("Display", selection: Binding(get: { model.display }, set: { model.select($0) })) {
                                ForEach(DisplayChoice.allCases) { Text($0.title).tag($0) }
                            }
                            if model.display.hasScreen {
                                Button("Resize Guest to Fit", systemImage: "arrow.up.left.and.arrow.down.right") {
                                    model.fitScreen(to: viewSize, scale: displayScale)
                                }
                                .disabled(!model.isRunning)
                            }
                        } label: {
                            Label("Machine", systemImage: "cpu")
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
