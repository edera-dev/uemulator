import SwiftUI
import UniformTypeIdentifiers

// A machine's settings: an edited copy of its definition, saved on Done. A machine's devices are
// fixed while it runs, so changes apply at its next start. Files picked here are copied into the
// machine's folder on Done, so Cancel leaves the folder as it was.
struct MachineSettingsView: View {
    @ObservedObject var model: MachineModel
    let onDelete: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var draft: MachineDefinition
    // `bootFrom` when the sheet opened. An unattended install that finishes while the sheet is
    // open switches the machine to `auto` (`settleBootstrap`); Done must not put the stale
    // `installer` back, or the next start runs the interactive installer over the installed
    // system. State, not a let: the view is re-made whenever the model changes.
    @State private var openedBootFrom: BootFrom
    @State private var engine: EngineKind
    @State private var confirmingDelete = false
    @State private var picking: Pick?
    // Picked, imported on Done.
    @State private var newISOs: [URL] = []
    @State private var newImages: [URL] = []
    @State private var newKernel: URL?
    @State private var newInitrd: URL?
    @State private var newHypervisor: URL?
    @State private var saving = false
    // Disks New Disk made in this sheet: the file exists from the moment it is added, so Cancel,
    // or Detach before Done, has to delete it or it stays behind in the machine's folder.
    @State private var createdDisks: [URL] = []
    @State private var reinstalling = false
    // Reinstall…'s answers, acted on once its sheet is gone: the reinstall saves these settings
    // first and closes this sheet too, and a sheet closed under its open child misbehaves.
    @State private var reinstallRequest: BootstrapRequest?

    enum Pick: Identifiable {
        case iso, image, kernel, initrd, hypervisor
        var id: Self { self }
    }

    private static let memoryChoices = [256, 512, 1024, 2048, 3072, 4096, 8192, 16384]
    private static let maxCPUs = 8
    private static let diskChoices = [8, 16, 32, 64, 128]
    // The web build's wasm32 memory: RAM past 3 GiB would start above 4 GiB (docs/plans/13-web.md).
    private static let webKitMaxMemoryMib = 3072

    init(model: MachineModel, onDelete: @escaping () -> Void) {
        self.model = model
        self.onDelete = onDelete
        _draft = State(initialValue: model.definition)
        _openedBootFrom = State(initialValue: model.definition.bootFrom)
        _engine = State(initialValue: model.kind)
    }

    var body: some View {
        NavigationStack {
            Form {
                if model.isRunning {
                    Label("Applies at the next start.", systemImage: "info.circle")
                        .foregroundStyle(.secondary)
                }
                Section("Hardware") {
                    Picker("Memory", selection: $draft.memoryMib) {
                        ForEach(memoryChoices, id: \.self) { Text(MachineDefinition.memoryTitle($0)).tag($0) }
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
                } footer: {
                    Text(engineFooter)
                }
                if let distro = draft.install { installSection(distro) }
                disksSection
                Section("Security") {
                    Toggle("TPM 2.0", isOn: $draft.tpm)
                    Toggle("Secure Boot", isOn: $draft.secureBoot)
                }
                advancedSection
                if model.suspended && !model.isRunning {
                    Section {
                        Button("Discard Suspended State", role: .destructive) { model.discardSuspended() }
                    } footer: {
                        // docs/plans/29-snapshots.md: the snapshot is the machine's devices' state,
                        // so a hardware change makes it unusable; either engine resumes it.
                        Text("Start resumes the machine where it was suspended, on either engine. " +
                             "Changing its memory, processors, display, network, disks or security " +
                             "makes the suspended state unusable until they are put back.")
                    }
                }
                Section {
                    Button("Delete Machine…", role: .destructive) { confirmingDelete = true }
                        .disabled(model.isRunning)
                }
            }
            .formStyle(.grouped)
            .disabled(saving)
            .overlay {
                if saving {
                    ProgressView("Importing…")
                        .padding(24)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
                }
            }
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
                    .disabled(saving)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { done() }.disabled(saving)
                }
            }
            .fileImporter(isPresented: Binding(get: { picking != nil }, set: { if !$0 { picking = nil } }),
                          allowedContentTypes: picking == .iso ? [UTType(filenameExtension: "iso") ?? .data, .data] : [.data]) { result in
                guard case .success(let url) = result, let target = picking else { return }
                switch target {
                case .iso:
                    newISOs.append(url)
                    if draft.memoryMib < MachineModel.cdromMemoryMib { draft.memoryMib = MachineModel.cdromMemoryMib }
                case .image: newImages.append(url)
                case .kernel: newKernel = url
                case .initrd: newInitrd = url
                case .hypervisor: newHypervisor = url
                }
            }
            .sheet(isPresented: $reinstalling, onDismiss: {
                guard let request = reinstallRequest else { return }
                reinstallRequest = nil
                done { model.reinstall(request) }
            }) {
                if let distro = draft.install {
                    BootstrapSheet(distro: distro, purpose: .reinstall, hostname: model.name, record: draft.bootstrap) { request in
                        reinstallRequest = request
                        return nil
                    }
                }
            }
            .confirmationDialog("Delete \(model.name)?", isPresented: $confirmingDelete, titleVisibility: .visible) {
                Button("Delete", role: .destructive) {
                    dismiss()
                    onDelete()
                }
            } message: {
                Text("Its disks and files are deleted.")
            }
        }
        // A swipe down would skip Cancel's cleanup of new disks.
        .interactiveDismissDisabled(!createdDisks.isEmpty || saving)
        #if os(macOS)
        .frame(minWidth: 480, minHeight: 600)
        #else
        .formSheetSizing()
        #endif
    }

    // An installer machine's: what it boots, its unattended install's answers, and Reinstall….
    private func installSection(_ distro: Distro) -> some View {
        Section {
            Picker("Boot From", selection: $draft.bootFrom) {
                ForEach(BootFrom.allCases) { Text($0.title).tag($0) }
            }
            if let record = draft.bootstrap { BootstrapRecordRows(record: record) }
            Button("Reinstall…", role: .destructive) { reinstalling = true }
                .disabled(model.isRunning || reinstallRefusal != nil)
        } header: {
            Text(distro.title)
        } footer: {
            Text(installFooter(distro))
        }
    }

    private func installFooter(_ distro: Distro) -> String {
        if model.isRunning { return "Stop the machine to reinstall it." }
        if let reinstallRefusal { return reinstallRefusal }
        return "Reinstall… erases the disk and installs \(distro.title) again without questions, then saves these settings and starts the machine."
    }

    // `MachineModel.reinstall` refuses these too, once the draft is saved; an ISO picked here is
    // imported in front of the disks, so it counts as the first disk already.
    private var reinstallRefusal: String? {
        if let iso = newISOs.first {
            return "The install erases the first disk, and \(iso.lastPathComponent), picked here, would be attached first and read-only: reinstall without it."
        }
        var saved = draft
        if let newHypervisor { saved.hypervisor = newHypervisor.lastPathComponent }
        return MachineModel.reinstallRefusal(saved)
    }

    private var disksSection: some View {
        Section("Disks") {
            if let cd = draft.cdrom.flatMap(MachineStore.resolve) {
                HStack {
                    Label(cd.lastPathComponent, systemImage: "opticaldisc")
                    Spacer()
                    Button("Eject", systemImage: "eject") { draft.cdrom = nil }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.borderless)
                }
            }
            ForEach(Array(draft.disks.enumerated()), id: \.offset) { index, disk in
                HStack {
                    Label(disk.url.lastPathComponent, systemImage: disk.readOnly ? "opticaldisc" : "internaldrive")
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Text(Self.fileSize(disk.url)).foregroundStyle(.secondary).monospacedDigit()
                }
                .swipeActions { Button("Detach", role: .destructive) { draft.disks.remove(at: index) } }
                .contextMenu { Button("Detach", role: .destructive) { draft.disks.remove(at: index) } }
            }
            ForEach(newISOs + newImages, id: \.self) { url in
                Label(url.lastPathComponent, systemImage: newISOs.contains(url) ? "opticaldisc" : "externaldrive")
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Menu("Add", systemImage: "plus") {
                Button("ISO…", systemImage: "opticaldisc") { picking = .iso }
                Button("Disk Image…", systemImage: "externaldrive") { picking = .image }
                Menu("New Disk", systemImage: "internaldrive") {
                    ForEach(Self.diskChoices, id: \.self) { gib in
                        Button("\(gib) GiB") { addDisk(sizeGib: gib) }
                    }
                }
            }
        }
    }

    private var advancedSection: some View {
        Section("Advanced") {
            Picker("Firmware", selection: $draft.firmware) {
                ForEach(FirmwareKind.allCases) { Text($0.title).tag($0) }
            }
            if draft.install == nil {
                fileRow("Kernel", systemImage: "cpu", current: draft.bootsKernel ? "kernel" : nil, picked: $newKernel,
                        choose: { picking = .kernel }, remove: { draft.bootsKernel = false })
                if draft.bootsKernel || newKernel != nil {
                    fileRow("Initrd", systemImage: "doc.zipper", current: nil, picked: $newInitrd,
                            choose: { picking = .initrd }, remove: nil)
                }
            }
            TextField("Command Line", text: $draft.cmdline, axis: .vertical)
                .autocorrectionDisabled()
                #if os(iOS)
                .textInputAutocapitalization(.never)
                #endif
                .font(.body.monospaced())
            Toggle("Replace Default Command Line", isOn: $draft.replaceCmdline)
            fileRow("Hypervisor", systemImage: "square.stack.3d.up", current: draft.hypervisor, picked: $newHypervisor,
                    choose: { picking = .hypervisor }, remove: { draft.hypervisor = nil })
            if draft.hypervisor != nil || newHypervisor != nil {
                TextField("Hypervisor Command Line", text: $draft.hypervisorCmdline, axis: .vertical)
                    .autocorrectionDisabled()
                    #if os(iOS)
                    .textInputAutocapitalization(.never)
                    #endif
                    .font(.body.monospaced())
            }
            Picker("Console", selection: $draft.serialConsole) {
                Text("Serial (ttyS0)").tag(true)
                Text("Virtio (hvc0)").tag(false)
            }
            Toggle("JIT", isOn: $draft.jit)
            Toggle("IOMMU", isOn: $draft.iommu)
            Toggle("SMT", isOn: $draft.smt)
            if engine == .native {
                Toggle("One Thread for All Processors", isOn: $draft.timeshare)
            }
            Toggle("Hardware RNG", isOn: $draft.rng)
            Toggle("Restart on Reboot", isOn: $draft.reboot)
        }
    }

    private func fileRow(_ title: String, systemImage: String, current: String?, picked: Binding<URL?>,
                         choose: @escaping () -> Void, remove: (() -> Void)?) -> some View {
        LabeledContent {
            HStack {
                Button(picked.wrappedValue?.lastPathComponent ?? current ?? "Choose…", action: choose)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if picked.wrappedValue != nil || (current != nil && remove != nil) {
                    Button("Remove", systemImage: "xmark.circle.fill") {
                        if picked.wrappedValue != nil { picked.wrappedValue = nil } else { remove?() }
                    }
                    .labelStyle(.iconOnly)
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                }
            }
        } label: {
            Label(title, systemImage: systemImage)
        }
    }

    // The WebKit engine cannot give a guest more than its wasm32 memory holds.
    private var memoryChoices: [Int] {
        var choices = engine == .webKit ? Self.memoryChoices.filter { $0 <= Self.webKitMaxMemoryMib } : Self.memoryChoices
        if !choices.contains(draft.memoryMib) { choices.append(draft.memoryMib) }
        return choices.sorted()
    }

    private var engineFooter: String {
        switch engine {
        case .native:
            #if os(iOS)
            "Interpreted: slower, but no memory limit and processors in parallel."
            #else
            "Native JIT."
            #endif
        case .webKit:
            "WebAssembly JIT in WebKit. Up to 3 GiB; processors take turns."
        }
    }

    // Saves the draft and imports the picked files, then closes the sheet and runs `after`, but
    // only if the import worked: `after` is Reinstall…, which would erase the disk on a
    // half-applied configuration, and its `start()` would clear the import's error.
    private func done(after: (() -> Void)? = nil) {
        removeCreatedDisks(keeping: draft.disks.map(\.url))
        // Named before the save, so its command line is saved with it; the file follows.
        if let newHypervisor { draft.hypervisor = newHypervisor.lastPathComponent }
        // Left alone here, so whatever the machine has now stands (see `openedBootFrom`).
        if draft.bootFrom == openedBootFrom { draft.bootFrom = model.definition.bootFrom }
        model.save(draft, engine: engine)
        let isos = newISOs, images = newImages, kernel = newKernel, initrd = newInitrd, hypervisor = newHypervisor
        guard !isos.isEmpty || !images.isEmpty || kernel != nil || initrd != nil || hypervisor != nil else {
            dismiss()
            after?()
            return
        }
        saving = true
        Task {
            var imported = true
            do {
                try await model.importFiles(isos: isos, images: images, kernel: kernel, initrd: initrd, hypervisor: hypervisor)
            } catch {
                model.problem = error.localizedDescription
                imported = false
            }
            saving = false
            dismiss()
            if imported { after?() }
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

    private static func fileSize(_ url: URL) -> String {
        let bytes = (try? url.resourceValues(forKeys: [.totalFileAllocatedSizeKey]).totalFileAllocatedSize)
            ?? (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}
