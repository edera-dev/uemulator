import SwiftUI
import UniformTypeIdentifiers

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

// The machine list and the selected machine (docs/plans/28-apple.md, M4).
//
// Launch arguments, for scripted runs (UserDefaults' argument domain reads them):
//   -machine NAME             select it
//   -autostart                start the selected machine
//   -new-machine NAME         create it first if it does not exist, from -new-source demo (the
//                             default), cdrom:FILE (in Documents) or empty, with a disk of
//                             -new-disk-mib MIB (default none), -new-memory-mib, -new-display,
//                             -new-network NO (a NIC unless NO)
//   -engine native|webkit, -debug-log FILE   see MachineModel
struct LibraryView: View {
    @StateObject private var store = MachineStore()
    @State private var selection: String?
    @State private var creating = false
    @State private var problem: String?
    @State private var deleting: String?
    // The machine a first launch gets, so the app opens on something that boots.
    private static let firstMachine = "alpine-demo"

    var body: some View {
        NavigationSplitView {
            List(store.names, id: \.self, selection: $selection) { name in
                MachineRow(model: store.model(name))
                    .contextMenu {
                        let model = store.model(name)
                        if model.isRunning {
                            Button("Stop", systemImage: "stop.fill") { model.stop() }
                        } else {
                            Button("Start", systemImage: "play.fill") {
                                selection = name
                                model.start()
                            }
                        }
                        Divider()
                        Button("Delete…", systemImage: "trash", role: .destructive) { deleting = name }
                            .disabled(model.isRunning)
                    }
            }
            .navigationTitle("Machines")
            .navigationSplitViewColumnWidth(min: 200, ideal: 240)
            .toolbar {
                ToolbarItem {
                    Button("New Machine", systemImage: "plus") { creating = true }
                }
            }
        } detail: {
            if let selection, store.names.contains(selection) {
                MachineView(model: store.model(selection)) { delete(selection) }
                    .id(selection)
            } else {
                ContentUnavailableView {
                    Label("No Machine", systemImage: "cpu")
                } description: {
                    Text("Pick a machine, or make one from a CD image or the Alpine demo.")
                } actions: {
                    Button("New Machine…") { creating = true }
                }
            }
        }
        .sheet(isPresented: $creating) {
            NewMachineSheet(store: store) { name in
                creating = false
                selection = name
            }
        }
        .confirmationDialog("Delete \(deleting ?? "")?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } }),
                            titleVisibility: .visible) {
            Button("Delete Machine and Its Disks", role: .destructive) {
                if let deleting { delete(deleting) }
            }
        } message: {
            Text("Its disks and NVRAM are deleted. A CD image it used stays where it is.")
        }
        .alert("Machines", isPresented: Binding(get: { problem != nil }, set: { if !$0 { problem = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(problem ?? "")
        }
        .task { launch() }
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
            let source: MachineStore.Source
            switch defaults.string(forKey: "new-source") ?? "demo" {
            case "empty": source = .empty
            case let value where value.hasPrefix("cdrom:"): source = .cdrom(.documents(String(value.dropFirst("cdrom:".count))))
            default: source = .demo
            }
            let display = DisplayChoice(rawValue: defaults.string(forKey: "new-display") ?? "") ?? .terminal
            let memory = defaults.integer(forKey: "new-memory-mib")
            let network = defaults.string(forKey: "new-network").map { !["no", "0", "false", "off"].contains($0) } ?? true
            do {
                try store.create(name, source: source, diskMib: UInt64(max(0, defaults.integer(forKey: "new-disk-mib"))),
                                 memoryMib: memory > 0 ? memory : MachineModel.demoMemoryMib, display: display, network: network)
            } catch {
                problem = error.localizedDescription
            }
        }
        if store.names.isEmpty {
            try? store.create(Self.firstMachine, source: .demo, diskMib: 0, memoryMib: MachineModel.demoMemoryMib, display: .terminal)
        }
        selection = defaults.string(forKey: "machine") ?? defaults.string(forKey: "new-machine") ?? store.names.first
        if ProcessInfo.processInfo.arguments.contains("-autostart"), let selection, store.names.contains(selection) {
            store.model(selection).start()
        }
    }
}

struct MachineRow: View {
    @ObservedObject var model: MachineModel

    var body: some View {
        Label {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.name).lineLimit(1)
                Text(summary).font(.caption).foregroundStyle(.secondary).lineLimit(1)
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
    }

    private var summary: String { Self.summary(model) }

    // What the machine is, in a line: memory, processors, what it boots.
    static func summary(_ model: MachineModel) -> String {
        let memory = model.definition.memoryMib
        var parts = [memory >= 1024 && memory % 1024 == 0 ? "\(memory / 1024) GiB" : "\(memory) MiB"]
        if model.definition.cpus > 1 { parts.append("\(model.definition.cpus) CPUs") }
        if let cd = model.cdromName { parts.append(cd) }
        if model.definition.bootsKernel { parts.append("kernel") }
        if !model.definition.disks.isEmpty { parts.append("\(model.definition.disks.count) disk\(model.definition.disks.count == 1 ? "" : "s")") }
        return parts.joined(separator: " · ")
    }
}

struct NewMachineSheet: View {
    enum Source: String, CaseIterable, Identifiable {
        case cdrom
        case demo
        case empty
        var id: String { rawValue }
        var title: String {
            switch self {
            case .cdrom: "CD image (installer or live ISO)"
            case .demo: "Alpine Linux demo"
            case .empty: "Nothing yet"
            }
        }
    }

    let store: MachineStore
    let onCreated: (String) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var source = Source.cdrom
    @State private var cdrom: URL?
    @State private var choosingISO = false
    @State private var memoryMib = MachineModel.cdromMemoryMib
    @State private var diskGib = 8
    @State private var display = DisplayChoice.screen
    @State private var network = true
    @State private var problem: String?

    private static let memoryChoices = [256, 512, 1024, 2048, 3072, 4096]
    private static let diskChoices = [0, 1, 4, 8, 16, 32, 64]
    var body: some View {
        NavigationStack {
            Form {
                TextField("Name", text: $name)
                    #if os(iOS)
                    .textInputAutocapitalization(.never)
                    #endif
                    .autocorrectionDisabled()
                Picker("Boots", selection: $source) {
                    ForEach(Source.allCases) { Text($0.title).tag($0) }
                }
                if source == .cdrom {
                    Button(cdrom?.lastPathComponent ?? "Choose CD Image…") { choosingISO = true }
                }
                Picker("Memory", selection: $memoryMib) {
                    ForEach(Self.memoryChoices, id: \.self) { Text($0 >= 1024 ? "\($0 / 1024) GiB" : "\($0) MiB").tag($0) }
                }
                Picker("Disk", selection: $diskGib) {
                    ForEach(Self.diskChoices, id: \.self) { Text($0 == 0 ? "None" : "\($0) GiB").tag($0) }
                }
                Picker("Display", selection: $display) {
                    ForEach(DisplayChoice.allCases) { Text($0.title).tag($0) }
                }
                Toggle("Network (internet through the app)", isOn: $network)
                if let problem {
                    Text(problem).foregroundStyle(.red)
                }
            }
            .formStyle(.grouped)
            .navigationTitle("New Machine")
            .fileImporter(isPresented: $choosingISO, allowedContentTypes: [UTType(filenameExtension: "iso") ?? .data, .data]) { result in
                if case .success(let url) = result { cdrom = url }
            }
            .onChange(of: source) {
                // The demo is a small initramfs on a serial console; CDs want room and a screen.
                memoryMib = source == .demo ? MachineModel.demoMemoryMib : MachineModel.cdromMemoryMib
                display = source == .demo ? .terminal : .screen
            }
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Create") { create() }
                        .disabled(name.isEmpty || (source == .cdrom && cdrom == nil))
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 420, minHeight: 360)
        #endif
    }

    private func create() {
        do {
            let machineSource: MachineStore.Source
            switch source {
            case .demo: machineSource = .demo
            case .empty: machineSource = .empty
            case .cdrom:
                guard let cdrom else { return }
                let scoped = cdrom.startAccessingSecurityScopedResource()
                defer { if scoped { cdrom.stopAccessingSecurityScopedResource() } }
                machineSource = .cdrom(.bookmark(try MachineStore.bookmark(for: cdrom)))
            }
            try store.create(name, source: machineSource, diskMib: UInt64(diskGib) * MachineStore.mibPerGib,
                             memoryMib: memoryMib, display: display, network: network)
            onCreated(name)
        } catch {
            problem = error.localizedDescription
        }
    }
}
