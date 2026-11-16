import SwiftUI
import UniformTypeIdentifiers
#if os(macOS)
import AppKit
#endif

@main
struct UemulatorApp: App {
    var body: some Scene {
        WindowGroup {
            LibraryView()
                #if os(macOS)
                .frame(minWidth: 720, minHeight: 480)
                #endif
        }
        #if os(macOS)
        .defaultSize(width: 1200, height: 800)
        #endif
    }
}

// The machine list and the selected machine (docs/plans/28-apple.md, "The UI as it is").
//
// Launch arguments, for scripted runs (UserDefaults' argument domain reads them):
//   -machine NAME             select it
//   -autostart                start the selected machine
//   -new-machine NAME         create it first if it does not exist, from -new-source demo (the
//                             default), demo:xen, install:alpine|debian|fedora, iso:FILE (in
//                             Documents, imported) or empty
//   -engine native|webkit, -debug-log FILE   see MachineModel
struct LibraryView: View {
    @StateObject private var store = MachineStore()
    @State private var selection: String?
    @State private var creating = false
    @State private var problem: String?
    @State private var deleting: String?
    // Only the screen or terminal: no sidebar, no bars (MachineView).
    @State private var fullscreen = false
    @State private var columns = NavigationSplitViewVisibility.automatic
    #if os(iOS)
    // A phone opens on the list, so an empty library offers the presets there; wider layouts show
    // them in the detail.
    @Environment(\.horizontalSizeClass) private var sizeClass
    private var compact: Bool { sizeClass == .compact }
    #else
    private let compact = false
    #endif

    var body: some View {
        NavigationSplitView(columnVisibility: $columns) {
            List(store.names, id: \.self, selection: $selection) { name in
                MachineRow(model: store.model(name), onStart: { selection = name }, onDelete: { deleting = name })
            }
            .overlay {
                if compact && store.names.isEmpty { EmptyLibrary(onDemo: add) }
            }
            .navigationTitle("uemulator")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .navigationSplitViewColumnWidth(min: 220, ideal: 260)
            .toolbar {
                ToolbarItem(placement: Self.brandPlacement) { BrandMark() }
                ToolbarItem(placement: .primaryAction) {
                    Menu {
                        NewMachineMenu(onDemo: add, onInstall: add) { creating = true }
                    } label: {
                        Label("New Machine", systemImage: "plus")
                    }
                    .help("New machine")
                }
            }
        } detail: {
            if let selection, store.names.contains(selection) {
                MachineView(model: store.model(selection), fullscreen: $fullscreen) { delete(selection) }
                    .id(selection)
            } else if store.names.isEmpty {
                EmptyLibrary(onDemo: add)
            } else {
                ContentUnavailableView("No Machine Selected", systemImage: "cpu")
            }
        }
        .onChange(of: fullscreen) {
            columns = fullscreen ? .detailOnly : .automatic
            #if os(macOS)
            // The window too, unless it already is (the user's green button).
            if let window = NSApp.keyWindow, window.styleMask.contains(.fullScreen) != fullscreen { window.toggleFullScreen(nil) }
            #endif
        }
        .onChange(of: selection) { fullscreen = false }
        .sheet(isPresented: $creating) {
            NewMachineSheet(store: store) { name in
                creating = false
                selection = name
            }
        }
        .confirmationDialog("Delete \(deleting ?? "")?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } }),
                            titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                if let deleting { delete(deleting) }
            }
        } message: {
            Text("Its disks and files are deleted.")
        }
        .alert("Machines", isPresented: Binding(get: { problem != nil }, set: { if !$0 { problem = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(problem ?? "")
        }
        .task { launch() }
    }

    // The logo and name lead the sidebar's bar: centered where a phone or iPad has room for it,
    // first in the toolbar on a Mac.
    #if os(iOS)
    private static let brandPlacement = ToolbarItemPlacement.principal
    #else
    private static let brandPlacement = ToolbarItemPlacement.navigation
    #endif

    private func add(_ demo: Demo) {
        make(demo.machineName, .demo(demo))
    }

    private func add(_ distro: Distro) {
        make(distro.rawValue, .install(distro))
    }

    // Demos and installs need nothing more from the user: made, selected and started.
    private func make(_ base: String, _ preset: MachineStore.Preset) {
        do {
            let name = try store.create(base, preset: preset)
            selection = name
            store.model(name).start()
        } catch {
            problem = error.localizedDescription
        }
    }

    private func delete(_ name: String) {
        do {
            try store.delete(name)
            selection = store.names.first
        } catch {
            problem = error.localizedDescription
        }
    }

    private func launch() {
        let defaults = UserDefaults.standard
        if let name = defaults.string(forKey: "new-machine"), !store.names.contains(name) {
            let source = defaults.string(forKey: "new-source") ?? "demo"
            do {
                switch source {
                case let value where value.hasPrefix("install:"):
                    guard let distro = Distro(rawValue: String(value.dropFirst("install:".count))) else { break }
                    try store.create(name, preset: .install(distro))
                case let value where value.hasPrefix("iso:"):
                    let iso = MachineStore.documents.appendingPathComponent(String(value.dropFirst("iso:".count)))
                    try store.create(name, preset: .custom(memoryMib: MachineModel.cdromMemoryMib, diskGib: Distro.easyDiskGib,
                                                           display: .screen, network: true))
                    Task {
                        do {
                            try await store.importDisk(name, from: iso, readOnly: true)
                            if ProcessInfo.processInfo.arguments.contains("-autostart") { store.model(name).start() }
                        } catch {
                            problem = error.localizedDescription
                        }
                    }
                case "empty":
                    try store.create(name, preset: .custom(memoryMib: MachineModel.cdromMemoryMib, diskGib: 0, display: .screen, network: true))
                default:
                    try store.create(name, preset: .demo(source == "demo:xen" ? .xen : .alpine))
                }
            } catch {
                problem = error.localizedDescription
            }
        }
        selection = defaults.string(forKey: "machine") ?? defaults.string(forKey: "new-machine") ?? store.names.first
        let importing = defaults.string(forKey: "new-source")?.hasPrefix("iso:") == true
        if ProcessInfo.processInfo.arguments.contains("-autostart"), !importing, let selection, store.names.contains(selection) {
            store.model(selection).start()
        }
    }
}

// The logo; a Mac's sidebar has room for the name beside it, a phone's bar does not.
struct BrandMark: View {
    var body: some View {
        HStack(spacing: 8) {
            Image("Logo")
                .resizable()
                .interpolation(.high)
                .frame(width: 24, height: 24)
            #if os(macOS)
            Text("uemulator").font(.headline)
            #endif
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("uemulator")
    }
}

// The Add menu: the easy installs, a machine set up by hand, and the demos.
struct NewMachineMenu: View {
    let onDemo: (Demo) -> Void
    let onInstall: (Distro) -> Void
    let onCustom: () -> Void

    var body: some View {
        Section("Install") {
            ForEach(Distro.allCases) { distro in Button(distro.title) { onInstall(distro) } }
        }
        Button("Custom…", systemImage: "slider.horizontal.3", action: onCustom)
        Menu("Demo", systemImage: "play.circle") {
            ForEach(Demo.allCases) { demo in Button(demo.title) { onDemo(demo) } }
        }
    }
}

// An empty library: the + button makes machines; a demo is one small menu away.
struct EmptyLibrary: View {
    let onDemo: (Demo) -> Void

    var body: some View {
        ContentUnavailableView {
            Label("No Machines", systemImage: "cpu")
        } actions: {
            Menu("Try a Demo", systemImage: "play.circle") {
                ForEach(Demo.allCases) { demo in Button(demo.title) { onDemo(demo) } }
            }
            .fixedSize()
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }
}

struct MachineRow: View {
    // Observed here, not in the list: the context menu must see the machine's current state.
    @ObservedObject var model: MachineModel
    let onStart: () -> Void
    let onDelete: () -> Void

    var body: some View {
        Label {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.name).lineLimit(1)
                Text(Self.summary(model)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        } icon: {
            Image(systemName: model.definition.display.hasScreen ? "display" : "terminal")
                .foregroundStyle(.secondary)
                .overlay(alignment: .bottomTrailing) {
                    if model.isRunning {
                        Circle().fill(.green).frame(width: 7, height: 7).offset(x: 3, y: 3)
                    }
                }
        }
        .padding(.vertical, 2)
        .contextMenu {
            if model.isRunning {
                Button("Stop", systemImage: "stop.fill") { model.stop() }
            } else {
                Button("Start", systemImage: "play.fill") {
                    onStart()
                    model.start()
                }
            }
            Divider()
            Button("Delete…", systemImage: "trash", role: .destructive, action: onDelete)
                .disabled(model.isRunning)
        }
    }

    // What the machine is, in a line: what it runs, then memory and processors.
    static func summary(_ model: MachineModel) -> String {
        var parts: [String] = []
        if let subject = model.subject { parts.append(subject) }
        parts.append(MachineDefinition.memoryTitle(model.definition.memoryMib))
        if model.definition.cpus > 1 { parts.append("\(model.definition.cpus) CPUs") }
        return parts.joined(separator: " · ")
    }
}

// A machine set up by hand: what it boots (each file is copied into the machine's folder), and its
// hardware. Everything else is in its settings.
struct NewMachineSheet: View {
    let store: MachineStore
    let onCreated: (String) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var iso: URL?
    @State private var diskImage: URL?
    @State private var kernel: URL?
    @State private var initrd: URL?
    @State private var picking: Pick?
    @State private var memoryMib = MachineModel.cdromMemoryMib
    @State private var diskGib = Distro.easyDiskGib
    @State private var display = DisplayChoice.screen
    @State private var network = true
    @State private var importing = false
    @State private var problem: String?

    enum Pick: Identifiable {
        case iso, disk, kernel, initrd
        var id: Self { self }
    }

    static let memoryChoices = [512, 1024, 2048, 3072, 4096, 8192]
    static let diskChoices = [0, 8, 16, 32, 64, 128]

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Name", text: $name)
                        #if os(iOS)
                        .textInputAutocapitalization(.never)
                        #endif
                        .autocorrectionDisabled()
                }
                Section("Boot") {
                    FileRow(title: "ISO", systemImage: "opticaldisc", file: $iso) { picking = .iso }
                    FileRow(title: "Disk Image", systemImage: "externaldrive", file: $diskImage) { picking = .disk }
                    FileRow(title: "Kernel", systemImage: "cpu", file: $kernel) { picking = .kernel }
                    if kernel != nil {
                        FileRow(title: "Initrd", systemImage: "doc.zipper", file: $initrd) { picking = .initrd }
                    }
                }
                Section("Hardware") {
                    Picker("Memory", selection: $memoryMib) {
                        ForEach(Self.memoryChoices, id: \.self) { Text(MachineDefinition.memoryTitle($0)).tag($0) }
                    }
                    Picker("New Disk", selection: $diskGib) {
                        ForEach(Self.diskChoices, id: \.self) { Text($0 == 0 ? "None" : "\($0) GiB").tag($0) }
                    }
                    Picker("Display", selection: $display) {
                        ForEach(DisplayChoice.allCases) { Text($0.title).tag($0) }
                    }
                    Toggle("Network", isOn: $network)
                }
                if let problem {
                    Section {
                        Label(problem, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
                    }
                }
            }
            .formStyle(.grouped)
            .disabled(importing)
            .overlay {
                if importing {
                    ProgressView("Importing…")
                        .padding(24)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
                }
            }
            .navigationTitle("New Machine")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .fileImporter(isPresented: Binding(get: { picking != nil }, set: { if !$0 { picking = nil } }),
                          allowedContentTypes: picking == .iso ? [UTType(filenameExtension: "iso") ?? .data, .data] : [.data]) { result in
                guard case .success(let url) = result, let target = picking else { return }
                switch target {
                case .iso: iso = url
                case .disk: diskImage = url
                case .kernel: kernel = url
                case .initrd: initrd = url
                }
                if name.isEmpty { name = Self.suggestedName(for: url) }
            }
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() }.disabled(importing) }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Create") { create() }
                        .disabled(name.isEmpty || importing)
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 440, minHeight: 440)
        #else
        .formSheetSizing()
        #endif
    }

    // "debian-13.1.0-amd64-netinst.iso" -> "debian-13.1.0-amd64-netinst"
    private static func suggestedName(for url: URL) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        let stem = url.deletingPathExtension().lastPathComponent
        let cleaned = String(stem.unicodeScalars.map { allowed.contains($0) && $0.isASCII ? Character($0) : "-" })
        return String(cleaned.prefix(48))
    }

    private func create() {
        if let invalid = MachineStore.validate(name) {
            problem = invalid
            return
        }
        let created: String
        do {
            created = try store.create(name, preset: .custom(memoryMib: memoryMib, diskGib: diskGib, display: display, network: network))
        } catch {
            problem = error.localizedDescription
            return
        }
        importing = true
        Task {
            do {
                if let iso { try await store.importDisk(created, from: iso, readOnly: true) }
                if let diskImage { try await store.importDisk(created, from: diskImage, readOnly: false) }
                if let kernel { try await store.importKernel(created, kernel: kernel, initrd: initrd) }
                importing = false
                onCreated(created)
            } catch {
                importing = false
                try? store.delete(created)
                problem = error.localizedDescription
            }
        }
    }
}

// A file to import: its name once chosen, with a button to clear it.
struct FileRow: View {
    let title: String
    let systemImage: String
    @Binding var file: URL?
    let choose: () -> Void

    var body: some View {
        LabeledContent {
            HStack {
                Button(file?.lastPathComponent ?? "Choose…", action: choose)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if file != nil {
                    Button("Clear", systemImage: "xmark.circle.fill") { file = nil }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                }
            }
        } label: {
            Label(title, systemImage: systemImage)
        }
    }
}

extension View {
    // What a machine runs, under its name in the window's title. iOS draws its own (MachineView's
    // principal item): the system's inline subtitle there is faint.
    @ViewBuilder
    func subtitle(_ text: String?) -> some View {
        #if os(macOS)
        navigationSubtitle(text ?? "")
        #else
        self
        #endif
    }

    // A form-sized sheet on iPad instead of a page sheet, where the OS has it (iOS 18).
    @ViewBuilder
    func formSheetSizing() -> some View {
        #if os(iOS)
        if #available(iOS 18.0, *) {
            presentationSizing(.form)
        } else {
            self
        }
        #else
        self
        #endif
    }
}
