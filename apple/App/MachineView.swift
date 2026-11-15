import SwiftUI
import UniformTypeIdentifiers
import WebKit

// One stored machine (MachineStore) and the engines that run it. The machine's devices are fixed
// when it starts: a change to its settings applies to the next start.
@MainActor
final class MachineModel: ObservableObject, ScreenInput {
    enum View: String, CaseIterable, Identifiable {
        case terminal
        case screen
        var id: String { rawValue }
    }

    let name: String
    private unowned let store: MachineStore

    @Published var state: EngineState = .stopped("")
    @Published var startedAt: Date?
    @Published private(set) var definition: MachineDefinition
    @Published var view: View = .terminal {
        // Keys held down while switching would never see their release.
        didSet { if view != oldValue { releaseAll() } }
    }
    @Published var problem: String?

    let terminal = TerminalSession()
    let screen = ScreenSurface()
    private let native = NativeEngine()
    private var webKitEngine: WebKitEngine?
    // A picked CD is readable only inside its security scope, held while the machine runs.
    private var scopedURL: URL?

    // Live ISOs want room: Fedora's 252 MB installer initramfs does not unpack in 1 GiB, and the
    // web page warns below 2 GiB for CDs (docs/plans/13-web.md).
    static let cdromMemoryMib = 2048
    static let demoMemoryMib = 256

    init(name: String, store: MachineStore) {
        self.name = name
        self.store = store
        definition = (try? store.load(name)) ?? MachineDefinition()
        view = definition.display == .screen ? .screen : .terminal
        wire(native)
        terminal.onInput = { [weak self] bytes in self?.engine.send(bytes) }
        terminal.onResize = { [weak self] cols, rows in self?.engine.resize(cols: cols, rows: rows) }
        openDebugLog()
    }

    // `-engine native|webkit` on the command line (UserDefaults' argument domain) wins over the
    // machine's choice, then the platform's default.
    var kind: EngineKind {
        EngineKind(rawValue: UserDefaults.standard.string(forKey: "engine") ?? "") ?? definition.engine ?? .platformDefault
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
        engine.onLog = { [weak self] text in
            self?.debugLog?.write(Data("\n[engine] \(text)\n".utf8))
        }
        engine.onState = { [weak self, weak engine] state in
            guard let self, let engine, engine === self.engine else { return }
            self.state = state
            if case .running = state { self.startedAt = Date() }
            if case .stopped = state { self.releaseScope() }
        }
    }

    var isRunning: Bool {
        switch state {
        case .starting, .running: true
        case .stopped: false
        }
    }

    var cdromName: String? {
        definition.cdrom.flatMap(MachineStore.resolve)?.lastPathComponent
    }

    private func update(_ change: (inout MachineDefinition) -> Void) {
        var changed = definition
        change(&changed)
        do {
            try store.save(changed, as: name)
            definition = changed
        } catch {
            problem = error.localizedDescription
        }
    }

    func select(_ newKind: EngineKind) {
        guard newKind != kind else { return }
        engine.stop()
        update { $0.engine = newKind }
        state = .stopped("")
    }

    func select(_ newDisplay: DisplayChoice) {
        guard newDisplay != definition.display else { return }
        update { $0.display = newDisplay }
        view = newDisplay == .screen ? .screen : .terminal
    }

    func insertCD(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            let bookmark = try MachineStore.bookmark(for: url)
            update {
                $0.cdrom = .bookmark(bookmark)
                if $0.memoryMib < Self.cdromMemoryMib { $0.memoryMib = Self.cdromMemoryMib }
            }
        } catch {
            problem = error.localizedDescription
        }
    }

    // After an install: the next start boots the disk.
    func ejectCD() {
        update { $0.cdrom = nil }
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

    func start() {
        problem = nil
        definition = (try? store.load(name)) ?? definition
        let size = terminal.size
        var settings = MachineSettings(
            kernel: Data(), initrd: Data(), disks: definition.disks, nvram: store.nvram(name),
            memoryMib: UInt32(clamping: definition.memoryMib), cpus: UInt32(clamping: definition.cpus),
            serialConsole: definition.serialConsole, display: definition.display,
            tpm: definition.tpm, secureBoot: definition.secureBoot,
            cols: UInt16(clamping: size.cols), rows: UInt16(clamping: size.rows))
        settings.cmdlineAppend = definition.cmdlineAppend
        if definition.firmware != "none" {
            settings.firmware = try? Data(contentsOf: Bundle.main.resourceURL!.appendingPathComponent("uemulator-firmware.bin"))
            guard settings.firmware != nil else {
                state = .stopped("The UEFI firmware is not in the app bundle; rebuild")
                return
            }
        }
        if definition.bootsKernel {
            guard let kernel = try? Data(contentsOf: store.kernel(name)) else {
                state = .stopped("The machine's kernel is missing (artifacts/kernel)")
                return
            }
            settings.kernel = kernel
            settings.initrd = (try? Data(contentsOf: store.initrd(name))) ?? Data()
        }
        if let cdrom = definition.cdrom {
            guard let url = MachineStore.resolve(cdrom) else {
                state = .stopped("The CD image cannot be found; insert it again")
                return
            }
            releaseScope()
            if url.startAccessingSecurityScopedResource() { scopedURL = url }
            // First: the firmware tries the disks in order, and the CD is what boots while it is in.
            settings.disks.insert(DiskImage(url: url, readOnly: true), at: 0)
        }
        guard !settings.kernel.isEmpty || !settings.disks.isEmpty else {
            state = .stopped("Nothing to boot: insert a CD")
            return
        }
        terminal.reset()
        engine.start(settings)
    }

    func stop() {
        engine.stop()
        releaseScope()
    }

    private func releaseScope() {
        scopedURL?.stopAccessingSecurityScopedResource()
        scopedURL = nil
    }

    // `-debug-log NAME` (in the app's Documents): the guest's console output, and every few
    // seconds a line saying what the screen is getting. For finding out, without the window,
    // whether a guest is stuck or its frames are not showing.
    private var debugLog: FileHandle?
    private var debugTimer: Timer?
    private static let debugInterval: TimeInterval = 5

    private func openDebugLog() {
        guard let file = UserDefaults.standard.string(forKey: "debug-log"), !file.isEmpty else { return }
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let url = documents.appendingPathComponent(file)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        debugLog = try? FileHandle(forWritingTo: url)
        let timer = Timer(timeInterval: Self.debugInterval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                let line = "\n[app] \(self.name) \(self.state) screen \(self.screen.width)x\(self.screen.height) frames \(self.screen.commits)\n"
                self.debugLog?.write(Data(line.utf8))
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        debugTimer = timer
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

    @ObservedObject var model: MachineModel
    let onDelete: () -> Void
    @State private var viewSize = CGSize.zero
    @State private var choosingISO = false
    @State private var softwareKeyboard = false
    @State private var confirmingDelete = false
    @Environment(\.displayScale) private var displayScale

    private var screenPane: ScreenPane {
        #if os(iOS)
        ScreenPane(surface: model.screen, input: model, showsSoftwareKeyboard: softwareKeyboard)
        #else
        ScreenPane(surface: model.screen, input: model)
        #endif
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Both stay alive so the terminal keeps its scrollback and the screen its image.
                TerminalPane(session: model.terminal)
                    .opacity(model.view == .terminal ? 1 : 0)
                    .allowsHitTesting(model.view == .terminal)
                if model.definition.display.hasScreen {
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
        .navigationTitle(model.name)
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .fileImporter(isPresented: $choosingISO, allowedContentTypes: [UTType(filenameExtension: "iso") ?? .data, .data]) { result in
            if case .success(let url) = result { model.insertCD(url) }
        }
        .confirmationDialog("Delete \(model.name)?", isPresented: $confirmingDelete, titleVisibility: .visible) {
            Button("Delete Machine and Its Disks", role: .destructive, action: onDelete)
        } message: {
            Text("Its disks and NVRAM are deleted. A CD image it used stays where it is.")
        }
        .alert("Machine", isPresented: Binding(get: { model.problem != nil }, set: { if !$0 { model.problem = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.problem ?? "")
        }
        .toolbar {
            // Toolbars show a Label's icon alone by default; the chip is mostly its text.
            ToolbarItem(placement: Self.statusPlacement) {
                StateChip(state: model.state, startedAt: model.startedAt).labelStyle(.titleAndIcon)
            }
            #if os(iOS)
            if model.definition.display.hasScreen && model.view == .screen {
                ToolbarItem(placement: Self.statusPlacement) {
                    Toggle(isOn: $softwareKeyboard) { Image(systemName: "keyboard") }
                        .toggleStyle(.button)
                }
            }
            #endif
            if model.definition.display.hasScreen {
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
                    if let cd = model.cdromName {
                        Button("Eject \(cd)", systemImage: "eject") { model.ejectCD() }
                    }
                    Button("Insert CD Image…", systemImage: "opticaldisc") { choosingISO = true }
                    Divider()
                    Picker("Engine", selection: Binding(get: { model.kind }, set: { model.select($0) })) {
                        ForEach(EngineKind.allCases) { Text($0.title).tag($0) }
                    }
                    Picker("Display", selection: Binding(get: { model.definition.display }, set: { model.select($0) })) {
                        ForEach(DisplayChoice.allCases) { Text($0.title).tag($0) }
                    }
                    if model.definition.display.hasScreen {
                        Button("Resize Guest to Fit", systemImage: "arrow.up.left.and.arrow.down.right") {
                            model.fitScreen(to: viewSize, scale: displayScale)
                        }
                        .disabled(!model.isRunning)
                    }
                    Divider()
                    Button("Delete Machine…", systemImage: "trash", role: .destructive) { confirmingDelete = true }
                        .disabled(model.isRunning)
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
