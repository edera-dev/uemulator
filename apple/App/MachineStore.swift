import Foundation
import UemulatorCore
#if os(iOS)
import UIKit
#endif

// Named machines in the app's Documents/Machines (docs/plans/28-apple.md), one folder each in
// the CLI's layout (`crates/uemulator-cli/src/machines.rs`):
//   machine.toml   the CLI's file; the app reads and writes only the keys below (Rust keeps the
//                  rest: ua_machine_load / ua_machine_save)
//   disk-<n>.udisk the machine's disks, sparse udisk images; imported ISOs and images beside them
//   nvram.kv       UEFI variables, Secure Boot keys and TPM state, on both engines (the WebKit
//                  engine's store writes it back through the loopback server)
//   artifacts/     an installer's kernel and initramfs under the CLI's names (`Distro`), or a
//                  kernel and initramfs to boot directly ("kernel", "initrd")
//   hypervisor     the image `hypervisor` names, as the CLI keeps it
//   app.json       what only the app knows: the engine, the demo. Not in machine.toml: the CLI
//                  drops keys it does not know when it saves.
//   suspended.snapshot  a suspended machine's whole state (docs/plans/29-snapshots.md), either
//                  engine's; deleted once the machine resumes from it
//   suspended.json what the disks looked like when it was taken (`SnapshotRecord`)
//
// The CLI writes absolute disk paths; an iOS app's container moves when the app is reinstalled,
// so a disk path that does not exist is looked for by name in the machine's own folder.

// A CD a machine made before imports existed still boots from: a file the user picked (a
// security-scoped bookmark) or one in the app's Documents. New machines import the file instead.
enum CDROM: Equatable {
    case bookmark(Data)
    case documents(String)
}

// The CLI's `boot_from`.
enum BootFrom: String, CaseIterable, Identifiable {
    // The disk once something is installed on it, else the installer.
    case auto
    case installer
    case disk

    var id: String { rawValue }

    var title: String {
        switch self {
        case .auto: "Automatic"
        case .installer: "Installer"
        case .disk: "Disk"
        }
    }
}

struct MachineDefinition {
    var memoryMib = 2048
    var cpus = 1
    var firmware = FirmwareKind.uefi
    var display = DisplayChoice.terminal
    var serialConsole = true
    var tpm = false
    var secureBoot = false
    var iommu = false
    var smt = true
    var timeshare = false
    var rng = true
    var jit = true
    // A guest reset boots the machine again; off, it stops.
    var reboot = true
    var cmdline = ""
    // `cmdline` replaces the default command line instead of adding to it.
    var replaceCmdline = false
    var disks: [DiskImage] = []
    // A NIC: the CLI's `[[networks]]`, kept as they were (the app only knows on or off).
    var networks: [[String: Any]] = []
    var network: Bool {
        get { !networks.isEmpty }
        set {
            if !newValue { networks = [] } else if networks.isEmpty { networks = [MachineStore.userNetwork] }
        }
    }
    // An installer machine: what it installs and whether it boots that or its disk.
    var install: Distro?
    var bootFrom = BootFrom.auto
    // The name of the file the hypervisor image came from (xen.gz, xen.efi); nil for none.
    var hypervisor: String?
    var hypervisorCmdline = ""
    // app.json
    var engine: EngineKind?
    var demo: Demo?
    var cdrom: CDROM?
    // artifacts/kernel is there: a directly booted kernel (not an installer's).
    var bootsKernel = false

    // How the UI writes a memory size: "512 MiB", "2 GiB".
    static func memoryTitle(_ mib: Int) -> String {
        let mibPerGib = Int(MachineStore.mibPerGib)
        return mib >= mibPerGib && mib % mibPerGib == 0 ? "\(mib / mibPerGib) GiB" : "\(mib) MiB"
    }
}

enum MachineStoreError: LocalizedError {
    case failed(String)
    var errorDescription: String? {
        switch self {
        case .failed(let message): message
        }
    }
}

@MainActor
final class MachineStore: ObservableObject {
    @Published private(set) var names: [String] = []
    let root: URL
    // One model per machine, kept while the app runs: a machine keeps running when another is
    // selected.
    private var models: [String: MachineModel] = [:]

    private static let machineFile = "machine.toml"
    private static let sidecarFile = "app.json"
    private static let nvramFile = "nvram.kv"
    private static let artifacts = "artifacts"
    private static let hypervisorFile = "hypervisor"
    private static let snapshotFile = "suspended.snapshot"
    private static let snapshotRecordFile = "suspended.json"
    private static let errorCapacity = 1024
    // The CLI's `--display` values for the app's choices.
    private static let displayNames: [DisplayChoice: String] = [.terminal: "serial", .screen: "framebuffer", .both: "both"]
    // What the CLI's machine names allow (machines.rs `validate_name`).
    private static let maxNameLength = 64
    // What `uemulator quick create` writes for its NAT NIC.
    nonisolated static let userNetwork: [String: Any] = ["model": "virtio", "link": "user"]
    nonisolated static let mibPerGib: UInt64 = 1024
    // The app's Documents: Machines/ is in it, and `-debug-log` and `cdrom:` files are read from it.
    nonisolated static let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!

    init() {
        root = Self.documents.appendingPathComponent("Machines")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        reload()
    }

    func reload() {
        let folders = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        names = folders
            .filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent(Self.machineFile).path) }
            .map(\.lastPathComponent)
            .sorted()
    }

    func directory(_ name: String) -> URL { root.appendingPathComponent(name) }
    func nvram(_ name: String) -> URL { directory(name).appendingPathComponent(Self.nvramFile) }
    func hypervisor(_ name: String) -> URL { directory(name).appendingPathComponent(Self.hypervisorFile) }
    func snapshot(_ name: String) -> URL { directory(name).appendingPathComponent(Self.snapshotFile) }
    private func snapshotRecord(_ name: String) -> URL { directory(name).appendingPathComponent(Self.snapshotRecordFile) }

    // A snapshot holds the guest mid-flight: its page cache, its file systems' journals, all of
    // which describe the disks as they were when it was taken. Resumed over disks that changed
    // since (booted from the CLI, say, or from the app with the snapshot set aside), the guest
    // would write through stale metadata and corrupt them. So the disks' sizes and modification
    // times are recorded once the suspend is complete (the library flushed them first, and a
    // flushed udisk writes nothing more when it closes) and must still be so to resume. Not
    // nvram.kv: every machine built from it compacts it (a resume that then fails included), so
    // its time says nothing, and only the CLI could change what is in it meanwhile.
    private struct SnapshotRecord: Codable, Equatable {
        struct File: Codable, Equatable {
            var path: String
            var size: UInt64
            var modified: Double
        }
        var disks: [File]
    }

    private static func describe(_ url: URL) -> SnapshotRecord.File? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = (attributes[.size] as? NSNumber)?.uint64Value,
              let modified = attributes[.modificationDate] as? Date
        else { return nil }
        return .init(path: url.lastPathComponent, size: size, modified: modified.timeIntervalSince1970)
    }

    private static func record(disks: [URL]) -> SnapshotRecord {
        SnapshotRecord(disks: disks.compactMap(describe))
    }

    func hasSnapshot(_ name: String) -> Bool {
        FileManager.default.fileExists(atPath: snapshot(name).path)
    }

    // Called once a suspend is complete, with the disks the machine ran with.
    func recordSnapshot(_ name: String, disks: [URL]) {
        let record = Self.record(disks: disks)
        if let data = try? JSONEncoder().encode(record) {
            try? data.write(to: snapshotRecord(name), options: .atomic)
        }
    }

    // Whether the snapshot may be resumed with these disks: a snapshot without its record (the
    // app died between the two) cannot be trusted either.
    func snapshotMatches(_ name: String, disks: [URL]) -> Bool {
        guard let data = try? Data(contentsOf: snapshotRecord(name)),
              let saved = try? JSONDecoder().decode(SnapshotRecord.self, from: data)
        else { return false }
        return saved == Self.record(disks: disks)
    }

    // The record goes with the snapshot: the engine deletes the snapshot itself once it resumes.
    func discardSnapshot(_ name: String) {
        try? FileManager.default.removeItem(at: snapshot(name))
        try? FileManager.default.removeItem(at: snapshotRecord(name))
    }

    func forgetSnapshotRecord(_ name: String) {
        try? FileManager.default.removeItem(at: snapshotRecord(name))
    }
    private func artifacts(_ name: String) -> URL { directory(name).appendingPathComponent(Self.artifacts) }

    // An installer's under the CLI's names, else the directly booted "kernel" and "initrd".
    func kernel(_ name: String, install: Distro? = nil) -> URL {
        artifacts(name).appendingPathComponent(install?.kernel.file ?? "kernel")
    }

    func initrd(_ name: String, install: Distro? = nil) -> URL {
        artifacts(name).appendingPathComponent(install?.initrd.file ?? "initrd")
    }

    // The display stays on while any machine runs or suspends (iOS): the idle timer locking the
    // phone would background the app, which suspends every running machine. App-wide rather
    // than per machine, so one machine stopping does not let the phone sleep under another.
    // Set again on every change and foregrounding: iOS does not promise to keep the flag.
    func updateKeepAwake() {
        #if os(iOS)
        let anyRunning = models.values.contains { $0.isRunning }
        if UIApplication.shared.isIdleTimerDisabled != anyRunning {
            UIApplication.shared.isIdleTimerDisabled = anyRunning
        }
        #endif
    }

    func model(_ name: String) -> MachineModel {
        if let model = models[name] { return model }
        let model = MachineModel(name: name, store: self)
        models[name] = model
        return model
    }

    static func validate(_ name: String) -> String? {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        guard let first = name.unicodeScalars.first, CharacterSet.alphanumerics.contains(first),
              name.count <= maxNameLength, name.unicodeScalars.allSatisfy(allowed.contains), name.allSatisfy(\.isASCII)
        else { return "Use letters, digits, '.', '_' or '-'" }
        return nil
    }

    // `base`, or `base-2`, `base-3`, ... whichever is free.
    func freeName(_ base: String) -> String {
        var name = base
        var number = 2
        while FileManager.default.fileExists(atPath: directory(name).path) {
            name = "\(base)-\(number)"
            number += 1
        }
        return name
    }

    func load(_ name: String) throws -> MachineDefinition {
        let folder = directory(name)
        var error = [CChar](repeating: 0, count: Self.errorCapacity)
        guard let json = ua_machine_load(folder.appendingPathComponent(Self.machineFile).path, &error, error.count) else {
            throw MachineStoreError.failed(String(cString: error))
        }
        defer { ua_string_free(json) }
        let table = (try? JSONSerialization.jsonObject(with: Data(String(cString: json).utf8))) as? [String: Any] ?? [:]
        var definition = MachineDefinition()
        if let memory = table["memory"] as? Int { definition.memoryMib = memory }
        if let cpus = table["cpus"] as? Int { definition.cpus = cpus }
        if let firmware = (table["firmware"] as? String).flatMap(FirmwareKind.init(rawValue:)) { definition.firmware = firmware }
        if let display = table["display"] as? String,
           let choice = Self.displayNames.first(where: { $0.value == display })?.key {
            definition.display = choice
        }
        // The CLI's default is one virtio console.
        if let consoles = table["consoles"] as? [String] { definition.serialConsole = consoles.first == "serial" }
        definition.tpm = table["tpm"] as? Bool ?? false
        definition.secureBoot = table["secure_boot"] as? Bool ?? false
        definition.iommu = table["iommu"] as? Bool ?? false
        definition.smt = table["smt"] as? Bool ?? true
        definition.timeshare = table["smp"] as? String == "timeshare"
        definition.rng = !(table["disable_rng"] as? Bool ?? false)
        definition.jit = !(table["disable_jit"] as? Bool ?? false)
        definition.reboot = !(table["no_reboot"] as? Bool ?? false)
        if let cmdline = table["cmdline"] as? String {
            definition.cmdline = cmdline
            definition.replaceCmdline = true
        } else {
            definition.cmdline = table["append"] as? String ?? ""
        }
        definition.install = (table["install"] as? String).flatMap(Distro.init(rawValue:))
        definition.bootFrom = (table["boot_from"] as? String).flatMap(BootFrom.init(rawValue:)) ?? .auto
        definition.hypervisor = table["hypervisor"] as? String
        definition.hypervisorCmdline = table["hypervisor_cmdline"] as? String ?? ""
        definition.networks = table["networks"] as? [[String: Any]] ?? []
        for disk in table["disks"] as? [[String: Any]] ?? [] {
            guard let path = disk["path"] as? String else { continue }
            var url = URL(fileURLWithPath: path)
            if !FileManager.default.fileExists(atPath: url.path) {
                url = folder.appendingPathComponent(url.lastPathComponent)
            }
            definition.disks.append(DiskImage(url: url, readOnly: disk["read_only"] as? Bool ?? false))
        }
        let sidecar = (try? JSONDecoder().decode(Sidecar.self, from: Data(contentsOf: folder.appendingPathComponent(Self.sidecarFile))))
            ?? Sidecar()
        definition.engine = sidecar.engine.flatMap(EngineKind.init(rawValue:))
        definition.demo = sidecar.demo.flatMap(Demo.init(rawValue:))
        definition.bootsKernel = FileManager.default.fileExists(atPath: kernel(name).path)
        if let bookmark = sidecar.cdromBookmark {
            definition.cdrom = .bookmark(bookmark)
        } else if let file = sidecar.cdromDocumentsFile {
            definition.cdrom = .documents(file)
        }
        return definition
    }

    func save(_ definition: MachineDefinition, as name: String) throws {
        let folder = directory(name)
        let optional: (String) -> Any = { $0.isEmpty ? NSNull() : $0 }
        let changes: [String: Any] = [
            "memory": definition.memoryMib,
            "cpus": definition.cpus,
            "firmware": definition.firmware.rawValue,
            "display": Self.displayNames[definition.display] ?? "serial",
            "consoles": [definition.serialConsole ? "serial" : "virtio"],
            "tpm": definition.tpm,
            "secure_boot": definition.secureBoot,
            "iommu": definition.iommu,
            "smt": definition.smt,
            "smp": definition.timeshare ? "timeshare" : NSNull(),
            "disable_rng": !definition.rng,
            "disable_jit": !definition.jit,
            "no_reboot": !definition.reboot,
            "cmdline": definition.replaceCmdline ? definition.cmdline : NSNull(),
            "append": definition.replaceCmdline ? NSNull() : optional(definition.cmdline),
            "install": definition.install?.rawValue ?? NSNull(),
            "boot_from": definition.install == nil ? NSNull() : definition.bootFrom.rawValue,
            "hypervisor": definition.hypervisor ?? NSNull(),
            "hypervisor_cmdline": definition.hypervisor == nil ? NSNull() : optional(definition.hypervisorCmdline),
            "disks": definition.disks.map { ["path": $0.url.path, "read_only": $0.readOnly] },
            "networks": definition.networks,
        ]
        let json = String(decoding: try JSONSerialization.data(withJSONObject: changes), as: UTF8.self)
        var error = [CChar](repeating: 0, count: Self.errorCapacity)
        guard ua_machine_save(folder.appendingPathComponent(Self.machineFile).path, json, &error, error.count) == 0 else {
            throw MachineStoreError.failed(String(cString: error))
        }
        var sidecar = Sidecar()
        sidecar.engine = definition.engine?.rawValue
        sidecar.demo = definition.demo?.rawValue
        switch definition.cdrom {
        case .bookmark(let data): sidecar.cdromBookmark = data
        case .documents(let file): sidecar.cdromDocumentsFile = file
        case nil: break
        }
        try JSONEncoder().encode(sidecar).write(to: folder.appendingPathComponent(Self.sidecarFile), options: .atomic)
    }

    enum Preset {
        case demo(Demo)
        // Easy mode: 2 GiB (Fedora 3), a 64 GiB disk, a NIC and the installer, downloaded on first
        // start.
        case install(Distro)
        // Hardware only: the caller imports what it boots (`importDisk`, `importKernel`).
        case custom(memoryMib: Int, diskGib: Int, display: DisplayChoice, network: Bool)
    }

    // Makes the machine's folder and machine.toml; returns its name (`base` or a free variant).
    @discardableResult
    func create(_ base: String, preset: Preset) throws -> String {
        let name = freeName(base)
        if let problem = Self.validate(name) { throw MachineStoreError.failed(problem) }
        let folder = directory(name)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        do {
            var definition = MachineDefinition()
            definition.network = true
            switch preset {
            case .demo(let demo):
                definition.demo = demo
                definition.memoryMib = Demo.memoryMib
                let resources = Demo.resources
                try FileManager.default.createDirectory(at: artifacts(name), withIntermediateDirectories: true)
                try FileManager.default.copyItem(at: resources.appendingPathComponent("vmlinuz-virt"), to: kernel(name))
                try FileManager.default.copyItem(at: resources.appendingPathComponent("initramfs.gz"), to: initrd(name))
                if demo == .xen, let manifest = demo.manifest, let image = manifest.hypervisor {
                    try FileManager.default.copyItem(at: resources.appendingPathComponent(image), to: hypervisor(name))
                    definition.hypervisor = image
                    definition.hypervisorCmdline = manifest.hypervisorArgs ?? ""
                    definition.iommu = manifest.iommu ?? false
                }
            case .install(let distro):
                definition.install = distro
                // The installer talks on the terminal (the console stays there), and the screen is
                // there for what the installed system draws. The demos stay terminal only.
                definition.display = .both
                definition.memoryMib = distro.memoryMib
                definition.disks = [try createDisk(name, sizeMib: UInt64(Distro.easyDiskGib) * Self.mibPerGib)]
            case .custom(let memoryMib, let diskGib, let display, let network):
                definition.memoryMib = memoryMib
                definition.display = display
                definition.network = network
                if diskGib > 0 { definition.disks = [try createDisk(name, sizeMib: UInt64(diskGib) * Self.mibPerGib)] }
            }
            try save(definition, as: name)
        } catch {
            try? FileManager.default.removeItem(at: folder)
            throw error
        }
        reload()
        return name
    }

    // Copies `source` into the machine's folder and attaches it: an ISO first and read-only (the
    // firmware tries the disks in order, and an installer must not see its medium as a target),
    // anything else last.
    func importDisk(_ name: String, from source: URL, readOnly: Bool) async throws {
        let copy = try await Importer.copy(source, into: directory(name))
        var definition = try load(name)
        let disk = DiskImage(url: copy, readOnly: readOnly)
        if readOnly { definition.disks.insert(disk, at: 0) } else { definition.disks.append(disk) }
        try save(definition, as: name)
        model(name).reload()
    }

    // A kernel (and initramfs) to boot directly, as artifacts/kernel and artifacts/initrd.
    func importKernel(_ name: String, kernel: URL, initrd: URL?) async throws {
        try FileManager.default.createDirectory(at: artifacts(name), withIntermediateDirectories: true)
        _ = try await Importer.copy(kernel, into: artifacts(name), as: "kernel")
        if let initrd { try await importInitrd(name, from: initrd) }
        model(name).reload()
    }

    func importInitrd(_ name: String, from source: URL) async throws {
        try FileManager.default.createDirectory(at: artifacts(name), withIntermediateDirectories: true)
        _ = try await Importer.copy(source, into: artifacts(name), as: "initrd")
    }

    func importHypervisor(_ name: String, from source: URL) async throws {
        _ = try await Importer.copy(source, into: directory(name), as: Self.hypervisorFile)
        var definition = try load(name)
        definition.hypervisor = source.lastPathComponent
        try save(definition, as: name)
        model(name).reload()
    }

    // The CLI's name for a machine's next disk: the first `disk-<n>.udisk` not taken.
    func createDisk(_ name: String, sizeMib: UInt64) throws -> DiskImage {
        let folder = directory(name)
        var number = 0
        while FileManager.default.fileExists(atPath: folder.appendingPathComponent("disk-\(number).udisk").path) { number += 1 }
        let disk = folder.appendingPathComponent("disk-\(number).udisk")
        var error = [CChar](repeating: 0, count: Self.errorCapacity)
        guard ua_disk_create(disk.path, sizeMib, &error, error.count) == 0 else {
            throw MachineStoreError.failed(String(cString: error))
        }
        return DiskImage(url: disk, readOnly: false)
    }

    static func diskInstalled(_ url: URL) -> Bool {
        var error = [CChar](repeating: 0, count: errorCapacity)
        return ua_disk_installed(url.path, &error, error.count) == 1
    }

    func removeKernel(_ name: String) throws {
        try? FileManager.default.removeItem(at: kernel(name))
        try? FileManager.default.removeItem(at: initrd(name))
    }

    func removeHypervisor(_ name: String) {
        try? FileManager.default.removeItem(at: hypervisor(name))
    }

    func delete(_ name: String) throws {
        models.removeValue(forKey: name)?.stop()
        updateKeepAwake()
        try FileManager.default.removeItem(at: directory(name))
        reload()
    }

    static func resolve(_ cdrom: CDROM) -> URL? {
        switch cdrom {
        case .documents(let file):
            return documents.appendingPathComponent(file)
        case .bookmark(let data):
            var stale = false
            #if os(macOS)
            return try? URL(resolvingBookmarkData: data, options: .withSecurityScope, relativeTo: nil, bookmarkDataIsStale: &stale)
            #else
            return try? URL(resolvingBookmarkData: data, options: [], relativeTo: nil, bookmarkDataIsStale: &stale)
            #endif
        }
    }

    private struct Sidecar: Codable {
        var engine: String?
        var demo: String?
        var cdromBookmark: Data?
        var cdromDocumentsFile: String?
    }
}
