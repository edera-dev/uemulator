import SwiftUI
import WebKit
#if os(iOS)
import UIKit
#endif

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
    // A snapshot waits in the machine's folder: the next start resumes it (docs/plans/29-snapshots.md).
    // Only meaningful while stopped; a run resuming from it deletes it once the guest runs.
    @Published private(set) var suspended = false
    // The disks the current run started with, recorded beside its snapshot once a suspend is complete.
    private var runningDisks: [URL] = []
    // Suspended because the app went to the background (iOS): the foreground resumes it.
    private var resumeOnForeground = false
    #if os(iOS)
    // Keeps the app running while a suspend writes the snapshot.
    private var suspendTask = UIBackgroundTaskIdentifier.invalid
    private var lifecycleObservers: [NSObjectProtocol] = []
    #endif

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
    // This run is an unattended install (`MachineStore.pendingBootstrap`): its reset is the
    // install's end, a success only once the console has shown the completion marker.
    private var bootstrapRun = false
    private var completionMarker = CompletionMarkerWatch()
    static let bootstrapFailedReason = "The installer reset before it finished. Use Reinstall… to try again."

    init(name: String, store: MachineStore) {
        self.name = name
        self.store = store
        definition = (try? store.load(name)) ?? MachineDefinition()
        view = definition.display == .screen ? .screen : .terminal
        suspended = store.hasSnapshot(name)
        wire(native)
        terminal.onInput = { [weak self] bytes in self?.engine.send(bytes) }
        terminal.onResize = { [weak self] cols, rows in self?.engine.resize(cols: cols, rows: rows) }
        openDebugLog()
        #if os(iOS)
        let center = NotificationCenter.default
        lifecycleObservers = [
            center.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.didEnterBackground() }
            },
            center.addObserver(forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.willEnterForeground() }
            },
        ]
        #endif
    }

    #if os(iOS)
    // iOS suspends the app soon and may end it (or only the WebKit engine's process) while it is
    // suspended: either way the machine would be gone. So it is suspended to its folder at once,
    // within the background grace, and resumed when the app is back; if iOS ended the app
    // meanwhile, the next start resumes it instead. At once rather than at the end of the
    // grace: writing a large guest's RAM can take a good part of it.
    private func didEnterBackground() {
        guard case .running = state else { return }
        resumeOnForeground = true
        suspend()
    }

    private func willEnterForeground() {
        store.updateKeepAwake()
        guard resumeOnForeground else { return }
        // A large guest may still be writing its snapshot when the user is back (unlocked at
        // once, say): `onState` resumes it once the suspend is complete.
        if isSuspending { return }
        resumeOnForeground = false
        // Only a finished suspend: one that failed says why and waits for the user.
        if case .stopped(EngineState.suspendedReason) = state { start() }
    }

    private func endSuspendTask() {
        guard suspendTask != .invalid else { return }
        UIApplication.shared.endBackgroundTask(suspendTask)
        suspendTask = .invalid
    }
    #endif

    // `-debug-suspend-after SECONDS`: suspends each run that long after it starts, so a scripted
    // launch (`-autostart`) exercises suspend and, launched again, resume.
    private func scheduleDebugSuspend() {
        let delay = UserDefaults.standard.double(forKey: "debug-suspend-after")
        guard delay > 0 else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self, case .running = self.state else { return }
            self.suspend()
        }
    }

    var isSuspending: Bool { state == EngineState.suspending }

    // What starting does: resume the suspended state, or boot.
    var startTitle: String { suspended ? "Resume" : "Start" }

    // Writes the machine's whole state to its folder and stops it; the next start resumes it.
    func suspend() {
        guard case .running = state else { return }
        #if os(iOS)
        if suspendTask == .invalid {
            suspendTask = UIApplication.shared.beginBackgroundTask(withName: "uemulator suspend") { [weak self] in
                MainActor.assumeIsolated { self?.endSuspendTask() }
            }
        }
        #endif
        engine.suspend()
    }

    // Forgets the suspended state: the next start boots afresh, as after a power cut.
    func discardSuspended() {
        guard !isRunning else { return }
        store.discardSnapshot(name)
        suspended = false
        if case .stopped(EngineState.suspendedReason) = state { state = .stopped("") }
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
            if self.bootstrapRun { self.completionMarker.feed(bytes) }
        }
        engine.onLog = { [weak self] text in
            self?.debugLog?.write(Data("\n[engine] \(text)\n".utf8))
        }
        engine.onState = { [weak self, weak engine] state in
            guard let self, let engine, engine === self.engine else { return }
            self.state = state
            self.store.updateKeepAwake()
            if case .running = state {
                self.startedAt = Date()
                self.scheduleDebugSuspend()
            }
            if case .stopped(let reason) = state {
                self.releaseScope()
                if reason == EngineState.suspendedReason {
                    self.store.recordSnapshot(self.name, disks: self.runningDisks)
                } else if reason.hasPrefix("Resume failed") {
                    // Kept: settings changed since the suspend can be put back. Settings can discard it.
                    self.problem = "\(reason). Put the machine's settings back as they were, or discard the suspended state in Settings."
                }
                self.suspended = self.store.hasSnapshot(self.name)
                #if os(iOS)
                self.endSuspendTask()
                // The background's suspend finished after the app came back (see
                // `willEnterForeground`). Not `.active`: between willEnterForeground and
                // didBecomeActive the app is `.inactive`, and the resume would be missed.
                if self.resumeOnForeground && reason == EngineState.suspendedReason
                    && UIApplication.shared.applicationState != .background {
                    self.resumeOnForeground = false
                    self.start()
                } else if reason != EngineState.suspendedReason {
                    self.resumeOnForeground = false
                }
                #endif
                // Before the restarts below: their `boot` decides afresh whether the next run is one.
                let wasBootstrapRun = self.bootstrapRun
                // A suspended install is still one when it resumes; any other end is not.
                if reason != EngineState.suspendedReason { self.bootstrapRun = false }
                if wasBootstrapRun && reason == EngineState.resetReason {
                    self.bootstrapReset()
                } else if reason == EngineState.resetReason && self.definition.reboot {
                    // Booted afresh, so an installer that rebooted comes up on its disk.
                    self.restartingAfterReset = true
                    self.start()
                }
            }
        }
    }

    // The end of an unattended install (docs/plans/35-quick-bootstrap.md, "After the install"),
    // whatever `reboot` says: the request is spent either way. After the marker the installed disk
    // boots (`auto` finds it installed; a reinstall's `installer` must not run the install
    // again). Without it, starting again would run an interactive installer nobody asked for.
    private func bootstrapReset() {
        store.pendingBootstrap[name] = nil
        guard completionMarker.seen else {
            state = .stopped(Self.bootstrapFailedReason)
            problem = Self.bootstrapFailedReason
            return
        }
        var changed = definition
        changed.bootFrom = .auto
        do {
            try store.save(changed, as: name)
        } catch {
            problem = error.localizedDescription
            return
        }
        restartingAfterReset = true
        start()
    }

    // Reinstall…: the disk is erased by an unattended install of the same distro, with fresh
    // installer images, booted at once as `quick reinstall` does. Only while stopped.
    func reinstall(_ request: BootstrapRequest) {
        guard !isRunning, let distro = definition.install else { return }
        // A suspended state describes the disk the install is about to erase.
        discardSuspended()
        var changed = definition
        changed.bootstrap = request.record
        // `auto` would boot the disk that is already installed.
        changed.bootFrom = .installer
        do {
            try store.save(changed, as: name)
        } catch {
            problem = error.localizedDescription
            return
        }
        store.removeInstaller(name, install: distro)
        store.pendingBootstrap[name] = request
        start()
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
        bootstrapRun = false
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
            if let request = store.pendingBootstrap[name] {
                // The answers follow the distro's own installer arguments (the bootstrap crate's
                // `Payload::cmdline`), before the user's.
                do {
                    let extended = try request.extend(settings.initrd, distro: distro)
                    settings.initrd = extended.initrd
                    cmdline.insert(extended.cmdline, at: 1)
                    bootstrapRun = true
                } catch {
                    state = .stopped("The install's answers: \(error.localizedDescription)")
                    return
                }
            }
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
        settings.suspendTo = store.snapshot(name)
        let disks = settings.disks.map(\.url)
        let resuming = store.hasSnapshot(name)
        if resuming && store.snapshotMatches(name, disks: disks) {
            settings.resumeFrom = store.snapshot(name)
        } else {
            if resuming {
                problem = "The suspended state was set aside: the disks changed after it was taken, and " +
                    "resuming over them would corrupt them. The machine boots afresh."
            }
            store.discardSnapshot(name)
            suspended = false
        }
        runningDisks = disks
        if settings.resumeFrom == nil {
            completionMarker = CompletionMarkerWatch()
        } else {
            // A resumed install is the same run, whatever its half-written disk makes `auto`
            // choose above, and may have shown the marker before its suspend.
            bootstrapRun = store.pendingBootstrap[name] != nil
        }
        // A resumed guest carries on where its console left off in this session.
        if !restartingAfterReset && settings.resumeFrom == nil { terminal.reset() }
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
                   folded: fold != nil, foldedBar: model.screenFoldedKeyBar)
        #else
        ScreenPane(surface: model.screen, input: model)
        #endif
    }

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
                        // Above the keyboard and its key bar (ScreenUIView's `contentRect`).
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
            if model.isRunning {
                ToolbarItem {
                    Button("Suspend", systemImage: "pause.fill") { model.suspend() }
                        .keyboardShortcut("s", modifiers: [.command, .shift])
                        .help("Suspend: save the machine's state and stop; Start resumes it")
                        .disabled(model.isSuspending)
                }
            }
            ToolbarItem {
                if model.isRunning {
                    Button("Stop", systemImage: "stop.fill") { model.stop() }
                        .keyboardShortcut(".", modifiers: .command)
                        .help("Stop")
                } else {
                    Button(model.startTitle, systemImage: "play.fill") { model.start() }
                        .keyboardShortcut("r", modifiers: .command)
                        .help(model.startTitle)
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
                Button(model.startTitle, systemImage: "play.fill") { model.start() }
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
    private static let normalStops: Set<String> = ["", "Stopped", "Powered off", "Reset", EngineState.suspendedReason]

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
