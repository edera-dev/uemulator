import SwiftUI
import UniformTypeIdentifiers

// A machine's settings: an edited copy of its definition, saved on Done. A machine's devices are
// fixed while it runs, so changes apply at its next start.
struct MachineSettingsView: View {
    @ObservedObject var model: MachineModel
    let onDelete: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var draft: MachineDefinition
    @State private var engine: EngineKind
    @State private var choosingISO = false
    @State private var confirmingDelete = false
    // Disks Add Disk made in this sheet: the file exists from the moment it is added, so Cancel,
    // or Detach before Done, has to delete it or it stays behind in the machine's folder.
    @State private var createdDisks: [URL] = []

    private static let memoryChoices = [256, 512, 1024, 2048, 3072, 4096, 8192]
    private static let maxCPUs = 8
    private static let diskChoices = [1, 4, 8, 16, 32, 64]
    // The web build's wasm32 memory: RAM past 3 GiB would start above 4 GiB (docs/plans/13-web.md).
    private static let webKitMaxMemoryMib = 3072

    init(model: MachineModel, onDelete: @escaping () -> Void) {
        self.model = model
        self.onDelete = onDelete
        _draft = State(initialValue: model.definition)
        _engine = State(initialValue: model.kind)
    }

    var body: some View {
        NavigationStack {
            Form {
                if model.isRunning {
                    Label("Changes apply the next time \(model.name) starts.", systemImage: "info.circle")
                        .foregroundStyle(.secondary)
                }
                Section("Hardware") {
                    Picker("Memory", selection: $draft.memoryMib) {
                        ForEach(memoryChoices, id: \.self) { Text(Self.memoryTitle($0)).tag($0) }
                    }
                    Stepper("Processors: \(draft.cpus)", value: $draft.cpus, in: 1...Self.maxCPUs)
                    Picker("Display", selection: $draft.display) {
                        ForEach(DisplayChoice.allCases) { Text($0.title).tag($0) }
                    }
                    Toggle("Network", isOn: $draft.network)
                }
                Section {
                    Picker("Engine", selection: $engine) {
                        ForEach(EngineKind.allCases) { Text($0.title).tag($0) }
                    }
                } header: {
                    Text("Engine")
                } footer: {
                    Text(engineFooter)
                }
                Section("Boot") {
                    if let cd = draft.cdrom.flatMap(MachineStore.resolve) {
                        LabeledContent("CD", value: cd.lastPathComponent)
                        Button("Eject CD", systemImage: "eject") { draft.cdrom = nil }
                    } else {
                        Button("Insert CD Image…", systemImage: "opticaldisc") { choosingISO = true }
                    }
                    if draft.bootsKernel {
                        LabeledContent("Kernel", value: "The Alpine demo's, booted first")
                        Button("Boot From the Disks Instead", systemImage: "internaldrive") { draft.bootsKernel = false }
                    }
                    TextField("Kernel command line additions", text: $draft.cmdlineAppend)
                        .autocorrectionDisabled()
                        #if os(iOS)
                        .textInputAutocapitalization(.never)
                        #endif
                }
                Section {
                    ForEach(Array(draft.disks.enumerated()), id: \.offset) { index, disk in
                        HStack {
                            Label(disk.url.lastPathComponent, systemImage: disk.readOnly ? "opticaldisc" : "internaldrive")
                            Spacer()
                            Text(Self.fileSize(disk.url)).foregroundStyle(.secondary).monospacedDigit()
                        }
                        .swipeActions { Button("Detach", role: .destructive) { draft.disks.remove(at: index) } }
                        .contextMenu { Button("Detach", role: .destructive) { draft.disks.remove(at: index) } }
                    }
                    Menu("Add Disk", systemImage: "plus") {
                        ForEach(Self.diskChoices, id: \.self) { gib in
                            Button("\(gib) GiB") { addDisk(sizeGib: gib) }
                        }
                    }
                } header: {
                    Text("Disks")
                } footer: {
                    Text("Sparse: a disk takes only what the guest has written. Detaching keeps its file in the machine's folder.")
                }
                Section("Security") {
                    Toggle("TPM 2.0", isOn: $draft.tpm)
                    Toggle("Secure Boot", isOn: $draft.secureBoot)
                }
                Section {
                    Button("Delete Machine…", role: .destructive) { confirmingDelete = true }
                        .disabled(model.isRunning)
                }
            }
            .formStyle(.grouped)
            .navigationTitle(model.name)
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        removeCreatedDisks(keeping: [])
                        dismiss()
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        removeCreatedDisks(keeping: draft.disks.map(\.url))
                        model.save(draft, engine: engine)
                        dismiss()
                    }
                }
            }
            .fileImporter(isPresented: $choosingISO, allowedContentTypes: [UTType(filenameExtension: "iso") ?? .data, .data]) { result in
                guard case .success(let url) = result, let cdrom = MachineModel.cdrom(for: url) else { return }
                draft.cdrom = cdrom
                if draft.memoryMib < MachineModel.cdromMemoryMib { draft.memoryMib = MachineModel.cdromMemoryMib }
            }
            .confirmationDialog("Delete \(model.name)?", isPresented: $confirmingDelete, titleVisibility: .visible) {
                Button("Delete Machine and Its Disks", role: .destructive) {
                    dismiss()
                    onDelete()
                }
            } message: {
                Text("Its disks and NVRAM are deleted. A CD image it used stays where it is.")
            }
        }
        // A swipe down would skip Cancel's cleanup of new disks.
        .interactiveDismissDisabled(!createdDisks.isEmpty)
        #if os(macOS)
        .frame(minWidth: 460, minHeight: 560)
        #endif
    }

    // The WebKit engine cannot give a guest more than its wasm32 memory holds.
    private var memoryChoices: [Int] {
        engine == .webKit ? Self.memoryChoices.filter { $0 <= Self.webKitMaxMemoryMib } : Self.memoryChoices
    }

    private var engineFooter: String {
        switch engine {
        case .native:
            #if os(iOS)
            "Runs in the app. iOS allows no JIT here, so the guest is interpreted: slow, but no memory cap and real processors in parallel."
            #else
            "Runs in the app with the native JIT: the fastest."
            #endif
        case .webKit:
            "Runs the WebAssembly build in WebKit, whose JIT iOS allows: several times faster than interpreting on iOS. Up to 3 GiB, processors take turns."
        }
    }

    private func addDisk(sizeGib: Int) {
        guard let disk = model.createDisk(sizeGib: sizeGib) else { return }
        createdDisks.append(disk.url)
        draft.disks.append(disk)
    }

    private func removeCreatedDisks(keeping kept: [URL]) {
        for url in createdDisks where !kept.contains(url) {
            try? FileManager.default.removeItem(at: url)
        }
        createdDisks = []
    }

    private static func memoryTitle(_ mib: Int) -> String {
        mib >= 1024 ? "\(mib / 1024) GiB" : "\(mib) MiB"
    }

    private static func fileSize(_ url: URL) -> String {
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}
