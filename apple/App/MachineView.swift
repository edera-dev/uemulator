import SwiftUI
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
    // Whether this machine has run since the app started: until then the view shows how to start it
    // rather than an empty terminal.
    @Published private(set) var hasStarted = false

    let terminal = TerminalSession()
    let screen = ScreenSurface()
    #if os(iOS)
    // The screen's keys below a foldable's fold: ScreenUIView sends them, FoldedControls shows them.
    let screenFoldedKeyBar = KeyAccessory(freestanding: true)
    #endif
    private let native = NativeEngine()
    private var webKitEngine: WebKitEngine?
    // A picked CD is readable only inside its security scope, held while the machine runs.
    private var scopedURL: URL?

    // Live ISOs want room: Fedora's 252 MB installer initramfs does not unpack in 1 GiB, and the
    // web page warns below 2 GiB for CDs (docs/plans/13-web.md).
    static let cdromMemoryMib = 2048
    // An installer's kernel and initramfs, fetched on the first start that needs them.
    private var preparing: Task<Void, Never>?
    // Set while the machine starts again after the guest reset: its terminal keeps what the last
    // boot printed (an installer's final screen, say).
    private var restartingAfterReset = false
    private let downloader = Downloader()

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
            if case .stopped(let reason) = state {
                self.releaseScope()
                // Booted afresh, so an installer that rebooted comes up on its disk.
                if reason == EngineState.resetReason && self.definition.reboot {
                    self.restartingAfterReset = true
                    self.start()
                }
            }
        }
    }

    var isRunning: Bool {
        switch state {
        case .preparing, .starting, .running: true
        case .stopped: false
        }
    }

    var cdromName: String? {
        definition.cdrom.flatMap(MachineStore.resolve)?.lastPathComponent
    }

    // What it runs, in a few words: the demo, the distro, the ISO.
    var subject: String? {
        if let demo = definition.demo { return demo.title }
        if let distro = definition.install { return distro.title }
        if let iso = definition.disks.first(where: \.readOnly) { return iso.url.lastPathComponent }
        return cdromName
    }

    func reload() {
        definition = (try? store.load(name)) ?? definition
    }

    // The CLI's `boot_from`: an installer machine boots its installer until something is on its
    // first disk.
    private var bootsInstaller: Bool {
        guard definition.install != nil else { return false }
        switch definition.bootFrom {
        case .installer: return true
        case .disk: return false
        case .auto: return !(definition.disks.first.map { MachineStore.diskInstalled($0.url) } ?? false)
        }
    }

    // The settings sheet's edits. The engine only changes while stopped: the engine object is the
    // running machine.
    func save(_ draft: MachineDefinition, engine newEngine: EngineKind) {
        var changed = draft
        if !isRunning { changed.engine = newEngine }
        do {
            // A machine made from the demo boots its kernel first, since the firmware always tries
            // a kernel it is given before the disks (docs/plans/19-uefi-firmware.md); booting the
            // disks means dropping it.
            if definition.bootsKernel && !changed.bootsKernel { try store.removeKernel(name) }
            if definition.hypervisor != nil && changed.hypervisor == nil { store.removeHypervisor(name) }
            try store.save(changed, as: name)
            let displayChanged = changed.display != definition.display
            definition = try store.load(name)
            if displayChanged { view = definition.display == .screen ? .screen : .terminal }
        } catch {
            problem = error.localizedDescription
        }
    }

    // The settings sheet's picked files, copied into the machine's folder.
    func importFiles(isos: [URL], images: [URL], kernel: URL?, initrd: URL?, hypervisor: URL?) async throws {
        for iso in isos { try await store.importDisk(name, from: iso, readOnly: true) }
        for image in images { try await store.importDisk(name, from: image, readOnly: false) }
        if let kernel {
            try await store.importKernel(name, kernel: kernel, initrd: initrd)
        } else if let initrd {
            try await store.importInitrd(name, from: initrd)
        }
        if let hypervisor { try await store.importHypervisor(name, from: hypervisor) }
        reload()
    }

    // A new sparse disk in the machine's folder, for the settings sheet to attach.
    func createDisk(sizeGib: Int) -> DiskImage? {
        do {
            return try store.createDisk(name, sizeMib: UInt64(sizeGib) * MachineStore.mibPerGib)
        } catch {
            problem = error.localizedDescription
            return nil
        }
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
        reload()
        if bootsInstaller, let distro = definition.install,
           [distro.kernel, distro.initrd].contains(where: { !FileManager.default.fileExists(atPath: artifact($0, distro).path) }) {
            hasStarted = true
            state = .preparing("Downloading")
            preparing = Task { [weak self] in
                guard let self else { return }
                do {
                    for file in [distro.kernel, distro.initrd] where !FileManager.default.fileExists(atPath: artifact(file, distro).path) {
                        try FileManager.default.createDirectory(at: artifact(file, distro).deletingLastPathComponent(),
                                                                withIntermediateDirectories: true)
                        let name = file.file
                        try await downloader.fetch(file.url, to: artifact(file, distro)) { [weak self] fraction in
                            Task { @MainActor in self?.downloaded(name, fraction) }
                        }
                    }
                    preparing = nil
                    boot()
                } catch {
                    preparing = nil
                    if !Task.isCancelled { state = .stopped(error.localizedDescription) }
                }
            }
            return
        }
        boot()
    }

    private func downloaded(_ file: String, _ fraction: Double) {
        guard case .preparing = state else { return }
        state = .preparing("Downloading \(file) \(Int(fraction * 100))%")
    }

    private func artifact(_ file: Distro.Download, _ distro: Distro) -> URL {
        file.file == distro.kernel.file ? store.kernel(name, install: distro) : store.initrd(name, install: distro)
    }

    private func boot() {
        let size = terminal.size
        var settings = MachineSettings(kernel: Data(), initrd: Data())
        settings.disks = definition.disks
        settings.nvram = store.nvram(name)
        settings.memoryMib = UInt32(clamping: definition.memoryMib)
        settings.cpus = UInt32(clamping: definition.cpus)
        settings.serialConsole = definition.serialConsole
        settings.jit = definition.jit
        settings.display = definition.display
        settings.tpm = definition.tpm
        settings.secureBoot = definition.secureBoot
        settings.iommu = definition.iommu
        settings.rng = definition.rng
        settings.smt = definition.smt
        settings.timeshare = definition.timeshare
        settings.network = definition.network
        settings.firmware = definition.firmware
        settings.cols = UInt16(clamping: size.cols)
        settings.rows = UInt16(clamping: size.rows)
        var cmdline = [definition.cmdline]
        if definition.firmware == .uefi {
            settings.firmwareImage = try? Data(contentsOf: Bundle.main.resourceURL!.appendingPathComponent("uemulator-firmware.bin"))
            guard settings.firmwareImage != nil else {
                state = .stopped("The UEFI firmware is not in the app bundle; rebuild")
                return
            }
        }
        if bootsInstaller, let distro = definition.install {
            guard let kernel = try? Data(contentsOf: store.kernel(name, install: distro)) else {
                state = .stopped("The installer's kernel is missing")
                return
            }
            settings.kernel = kernel
            settings.initrd = (try? Data(contentsOf: store.initrd(name, install: distro))) ?? Data()
            cmdline.insert(distro.cmdline, at: 0)
        } else if definition.install == nil && definition.bootsKernel {
            guard let kernel = try? Data(contentsOf: store.kernel(name)) else {
                state = .stopped("The kernel is missing")
                return
            }
            settings.kernel = kernel
            settings.initrd = (try? Data(contentsOf: store.initrd(name))) ?? Data()
        }
        if definition.demo != nil { cmdline.append(Demo.hostArgument) }
        settings.cmdlineAppend = cmdline.filter { !$0.isEmpty }.joined(separator: " ")
        settings.replaceCmdline = definition.replaceCmdline
        // A hypervisor boots the kernel as its dom0; an installed disk's boot loader cannot start
        // under it, so the disk boots without (the CLI refuses that combination instead).
        if definition.hypervisor != nil && !settings.kernel.isEmpty {
            guard let image = try? Data(contentsOf: store.hypervisor(name)) else {
                state = .stopped("The hypervisor image is missing")
                return
            }
            settings.hypervisor = image
            settings.hypervisorCmdline = definition.hypervisorCmdline
        }
        if let cdrom = definition.cdrom {
            guard let url = MachineStore.resolve(cdrom) else {
                state = .stopped("The ISO cannot be found")
                return
            }
            releaseScope()
            if url.startAccessingSecurityScopedResource() { scopedURL = url }
            // First: the firmware tries the disks in order, and the CD is what boots while it is in.
            settings.disks.insert(DiskImage(url: url, readOnly: true), at: 0)
        }
        guard !settings.kernel.isEmpty || !settings.disks.isEmpty else {
            state = .stopped("Nothing to boot")
            return
        }
        if !restartingAfterReset { terminal.reset() }
        restartingAfterReset = false
        hasStarted = true
        engine.start(settings)
    }

    func stop() {
        if let preparing {
            preparing.cancel()
            self.preparing = nil
            state = .stopped("Stopped")
            return
        }
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
    private static let debugInterval: TimeInterval = 5

    private func openDebugLog() {
        let defaults = UserDefaults.standard
        guard let file = defaults.string(forKey: "debug-log"), !file.isEmpty else { return }
        // Only the machine the launch named: every model is built at launch, and `createFile`
        // replaces the file, so a second model would leave the first writing to an unlinked one.
        let named = defaults.string(forKey: "machine") ?? defaults.string(forKey: "new-machine")
        if let named, named != name { return }
        let url = MachineStore.documents.appendingPathComponent(file)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        debugLog = try? FileHandle(forWritingTo: url)
        let timer = Timer(timeInterval: Self.debugInterval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                let line = "\n[app] \(self.name) \(self.state) screen \(self.screen.width)x\(self.screen.height) frames \(self.screen.commits)\n"
                self.debugLog?.write(Data(line.utf8))
            }
        }
        // The run loop keeps it; it lives as long as the app, as the model does.
        RunLoop.main.add(timer, forMode: .common)
    }
}

struct MachineView: View {
    // A phone's navigation bar has room for the title and a few buttons: the Terminal | Screen
    // switch and the screen's buttons go to the bottom bar there, and the state is behind an info
    // button in the top bar (`MachineInfoButton`) instead of a chip.
    #if os(iOS)
    private static let statusPlacement = ToolbarItemPlacement.bottomBar
    #else
    private static let statusPlacement = ToolbarItemPlacement.automatic
    #endif

    @ObservedObject var model: MachineModel
    @Binding var fullscreen: Bool
    let onDelete: () -> Void
    @State private var viewSize = CGSize.zero
    // Duo unfolded in portrait: the screen above the software keyboard instead of under it.
    @AppStorage("keyboardBelowScreen") private var keyboardBelowScreen = false
    @State private var softwareKeyboard = false
    @State private var editingSettings = false
    // Partially open in portrait (FoldedLayout.swift): the guest above the fold, the controls below.
    @State private var fold: FoldPosture?
    // Full screen came from folding, so leaving the fold ends it; one the user asked for stays.
    @State private var fullscreenFromFold = false
    @Environment(\.displayScale) private var displayScale

    private var screenPane: ScreenPane {
        #if os(iOS)
        ScreenPane(surface: model.screen, input: model, showsSoftwareKeyboard: $softwareKeyboard,
                   keepsAboveKeyboard: splitKeyboard || fold != nil, folded: fold != nil, foldedBar: model.screenFoldedKeyBar)
        #else
        ScreenPane(surface: model.screen, input: model)
        #endif
    }

    // A phone held upright whose screen is nearly square: a foldable unfolded. Measured without
    // the keyboard, which would otherwise make it look landscape once shown.
    private var unfoldedPortrait: Bool {
        #if os(iOS)
        UIDevice.current.userInterfaceIdiom == .phone && viewSize.height > viewSize.width
            && viewSize.width / viewSize.height >= Self.unfoldedAspect
        #else
        false
        #endif
    }

    // Width over height above any slab phone's (about 0.46) and below a foldable's inner screen.
    private static let unfoldedAspect: CGFloat = 0.6

    private var splitKeyboard: Bool { keyboardBelowScreen && unfoldedPortrait && model.view == .screen }

    // The guest's part of the view: above the fold, short of it by the clearance, when folded; else
    // all of it.
    private var guestHeight: CGFloat? { fold.map { $0.foldY - FoldPosture.clearance } }
    private var guestSize: CGSize {
        guestHeight.map { CGSize(width: viewSize.width, height: $0) } ?? viewSize
    }

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                ZStack {
                    // Both stay alive so the terminal keeps its scrollback and the screen its image.
                    TerminalPane(session: model.terminal)
                        .opacity(model.view == .terminal ? 1 : 0)
                        .allowsHitTesting(model.view == .terminal)
                    if model.definition.display.hasScreen {
                        // Under the keyboard by default; above it when split (ScreenUIView's
                        // `keepsAboveKeyboard`).
                        screenPane
                            .opacity(model.view == .screen ? 1 : 0)
                            .allowsHitTesting(model.view == .screen)
                    }
                    if !model.hasStarted && !model.isRunning {
                        StoppedPlaceholder(model: model) { editingSettings = true }
                    }
                    if case .preparing(let what) = model.state {
                        VStack(spacing: 12) {
                            ProgressView()
                            Text(what).monospacedDigit().foregroundStyle(.secondary)
                        }
                        .padding(24)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
                    }
                }
                // The ZStack stays the first child either way, so folding keeps the panes' views.
                .frame(height: guestHeight)
                #if os(iOS)
                if fold != nil {
                    FoldedControls(model: model, fullscreen: $fullscreen, softwareKeyboard: $softwareKeyboard,
                                   screenSize: guestSize) { editingSettings = true }
                }
                #endif
            }
            #if os(iOS)
            .background { HingeReader(posture: $fold) }
            #endif
            .onAppear { viewSize = geometry.size }
            .onChange(of: geometry.size) { viewSize = geometry.size }
        }
        .ignoresSafeArea(.keyboard)
        // Turning the split on brings the keyboard up; a layout change (full screen, rotating) does not.
        .onChange(of: keyboardBelowScreen) { if keyboardBelowScreen { softwareKeyboard = true } }
        #if os(iOS)
        // Folded upright is full screen; folding flat (or turning sideways) ends it. Set on a change,
        // not derived from the fold, so the Full Screen button and leaving the machine still work;
        // `initial` makes a machine opened while folded start full screen.
        .onChange(of: fold != nil, initial: true) {
            let folded = fold != nil
            model.terminal.folded = folded
            if folded {
                if !fullscreen { fullscreenFromFold = true }
                fullscreen = true
            } else if fullscreenFromFold {
                fullscreenFromFold = false
                fullscreen = false
            }
        }
        .onChange(of: fullscreen) { if !fullscreen { fullscreenFromFold = false } }
        // The screen's keyboard would go on typing into the hidden screen, and folded the bar on
        // show is the terminal's, whose hide key cannot put it away.
        .onChange(of: model.view) { if model.view != .screen { softwareKeyboard = false } }
        #endif
        .background(Color.black)
        // The WebKit engine's web view: in the hierarchy but invisible, never the UI.
        .background {
            if model.kind == .webKit {
                HiddenWebView(webView: model.webKit.webView).frame(width: 1, height: 1).opacity(0)
            }
        }
        .navigationTitle(model.name)
        .subtitle(model.subject)
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar(fullscreen ? .hidden : .automatic, for: .navigationBar)
        // Folded, the bottom bar's buttons are in FoldedControls.
        .toolbar(fullscreen || fold != nil ? .hidden : .automatic, for: .bottomBar)
        // The status bar is hidden by LibraryView: set here, in a split view's detail, iOS ignores it.
        #else
        .toolbar(fullscreen ? .hidden : .automatic, for: .windowToolbar)
        #endif
        .ignoresSafeArea(edges: fullscreen ? .all : [])
        // After the line above: the screen goes edge to edge, the buttons stay in the safe area, at
        // the bottom with the keyboard. Folded, FoldedControls has them.
        .overlay(alignment: .bottomTrailing) {
            if fullscreen && fold == nil {
                HStack(spacing: 12) {
                    #if os(iOS)
                    if model.view == .screen {
                        Toggle(isOn: $softwareKeyboard) { Label("Keyboard", systemImage: "keyboard") }
                            .toggleStyle(.button)
                    }
                    #endif
                    Button("Exit Full Screen", systemImage: "arrow.down.right.and.arrow.up.left") { fullscreen = false }
                        .keyboardShortcut("f", modifiers: [.command, .control])
                }
                .labelStyle(.iconOnly)
                .padding(10)
                .background(.ultraThinMaterial, in: Capsule())
                .padding(12)
            }
        }
        .sheet(isPresented: $editingSettings) {
            MachineSettingsView(model: model, onDelete: onDelete)
        }
        .alert("Machine", isPresented: Binding(get: { model.problem != nil }, set: { if !$0 { model.problem = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.problem ?? "")
        }
        .toolbar {
            #if os(iOS)
            // The system's inline title fades a subtitle out; the name and what it runs stay legible.
            ToolbarItem(placement: .principal) {
                VStack(spacing: 0) {
                    Text(model.name).font(.headline)
                    if let subject = model.subject {
                        Text(subject).font(.caption).foregroundStyle(.secondary)
                    }
                }
                .lineLimit(1)
            }
            ToolbarItem {
                MachineInfoButton(model: model)
            }
            #else
            // Toolbars show a Label's icon alone by default; the chip is mostly its text.
            ToolbarItem(placement: Self.statusPlacement) {
                StateChip(state: model.state, startedAt: model.startedAt).labelStyle(.titleAndIcon)
            }
            #endif
            if model.definition.display.hasScreen {
                ToolbarItem(placement: Self.statusPlacement) {
                    Picker("View", selection: $model.view) {
                        Label("Terminal", systemImage: "terminal").tag(MachineModel.View.terminal)
                        Label("Screen", systemImage: "display").tag(MachineModel.View.screen)
                    }
                    .pickerStyle(.segmented)
                    .labelStyle(.iconOnly)
                    .help("Terminal or screen")
                }
                if model.view == .screen {
                    #if os(iOS)
                    ToolbarItem(placement: Self.statusPlacement) {
                        Toggle(isOn: $softwareKeyboard) { Label("Keyboard", systemImage: "keyboard") }
                            .toggleStyle(.button)
                    }
                    if unfoldedPortrait {
                        ToolbarItem(placement: Self.statusPlacement) {
                            Toggle(isOn: $keyboardBelowScreen) { Label("Keyboard Below Screen", systemImage: "rectangle.split.1x2") }
                                .toggleStyle(.button)
                        }
                    }
                    #endif
                    ToolbarItem(placement: Self.statusPlacement) {
                        Button("Resize to Fit", systemImage: "aspectratio") {
                            model.fitScreen(to: guestSize, scale: displayScale)
                        }
                        .disabled(!model.isRunning)
                        .help("Ask the guest for a screen the size of the window")
                    }
                }
            }
            ToolbarItem(placement: Self.statusPlacement) {
                Button("Full Screen", systemImage: "arrow.up.left.and.arrow.down.right") { fullscreen = true }
                    .keyboardShortcut("f", modifiers: [.command, .control])
                    .help("Full screen")
            }
            ToolbarItem {
                Button("Settings", systemImage: "slider.horizontal.3") { editingSettings = true }
                    .keyboardShortcut("i", modifiers: .command)
                    .help("Settings")
            }
            ToolbarItem {
                if model.isRunning {
                    Button("Stop", systemImage: "stop.fill") { model.stop() }
                        .keyboardShortcut(".", modifiers: .command)
                        .help("Stop")
                } else {
                    Button("Start", systemImage: "play.fill") { model.start() }
                        .keyboardShortcut("r", modifiers: .command)
                        .help("Start")
                }
            }
        }
    }
}

// What a machine shows before its first start since the app opened: what it boots and how to start
// it, instead of an empty black terminal.
struct StoppedPlaceholder: View {
    @ObservedObject var model: MachineModel
    let onSettings: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "cpu").font(.system(size: 48)).foregroundStyle(.secondary)
            Text(model.name).font(.title2.bold())
            Text(MachineRow.summary(model)).foregroundStyle(.secondary)
            if case .stopped(let reason) = model.state, !reason.isEmpty {
                Label(reason, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
            }
            HStack {
                Button("Settings…", systemImage: "slider.horizontal.3", action: onSettings)
                    .buttonStyle(.bordered)
                Button("Start", systemImage: "play.fill") { model.start() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
            .controlSize(.large)
        }
        .multilineTextAlignment(.center)
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background)
    }
}

#if os(iOS)
// The top bar's info button: its icon says starting, running or failed at a glance, and it opens
// the state chip with the engine and the machine's summary.
struct MachineInfoButton: View {
    @ObservedObject var model: MachineModel
    @State private var showing = false
    // Ends that are not failures: the user's Stop, the guest's poweroff or reboot, and never
    // started. Any other reason is an error the icon flags until the popover is read.
    private static let normalStops: Set<String> = ["", "Stopped", "Powered off", "Reset"]

    var body: some View {
        Button { showing = true } label: {
            Label("Info", systemImage: icon)
        }
        // The glass toolbar ignores a label's foreground style; it takes the button's tint.
        .tint(tint)
        .help("State, uptime and engine")
        .popover(isPresented: $showing) {
            VStack(alignment: .leading, spacing: 10) {
                StateChip(state: model.state, startedAt: model.startedAt)
                    .labelStyle(.titleAndIcon)
                    .font(.headline)
                    .fixedSize(horizontal: false, vertical: true)
                LabeledContent("Engine", value: model.kind.title)
                LabeledContent("Machine", value: MachineRow.summary(model))
            }
            .padding()
            .frame(minWidth: 280, alignment: .leading)
            // A phone would otherwise make it a full sheet.
            .presentationCompactAdaptation(.popover)
        }
    }

    private var icon: String {
        switch model.state {
        case .preparing, .starting: "hourglass"
        case .running: "info.circle.fill"
        case .stopped(let reason): Self.normalStops.contains(reason) ? "info.circle" : "exclamationmark.circle.fill"
        }
    }

    private var tint: Color {
        switch model.state {
        case .preparing, .starting: .secondary
        case .running: .green
        case .stopped(let reason): Self.normalStops.contains(reason) ? .primary : .orange
        }
    }
}
#endif

// The web page's state chip: the uptime and how the machine runs, or why it stopped.
struct StateChip: View {
    let state: EngineState
    let startedAt: Date?

    var body: some View {
        switch state {
        case .preparing(let what):
            Label(what, systemImage: "arrow.down.circle").monospacedDigit()
        case .starting:
            Label("Starting", systemImage: "hourglass")
        case .running(let detail):
            TimelineView(.periodic(from: .now, by: 1)) { context in
                // The green dot says running; the text stays short enough for a phone's toolbar.
                Label("\(Self.uptime(since: startedAt, at: context.date)) · \(detail)", systemImage: "circle.fill")
                    .foregroundStyle(.green)
                    .monospacedDigit()
            }
        case .stopped(let reason):
            Label(reason.isEmpty ? "Stopped" : reason, systemImage: "stop.circle")
                .foregroundStyle(.secondary)
        }
    }

    // "4:05", or "1:04:05" past an hour.
    private static func uptime(since start: Date?, at now: Date) -> String {
        let secondsPerMinute = 60, minutesPerHour = 60
        let seconds = max(0, Int(now.timeIntervalSince(start ?? now)))
        let minutes = seconds / secondsPerMinute
        let hours = minutes / minutesPerHour
        let rest = String(format: "%02d", seconds % secondsPerMinute)
        return hours > 0 ? "\(hours):\(String(format: "%02d", minutes % minutesPerHour)):\(rest)" : "\(minutes):\(rest)"
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
