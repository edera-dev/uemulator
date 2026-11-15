import Foundation
import UemulatorCore

// Named machines in the app's Documents/Machines (docs/plans/28-apple.md, M4), one folder each in
// the CLI's layout (`crates/uemulator-cli/src/machines.rs`):
//   machine.toml   the CLI's file; the app reads and writes only the keys below (Rust keeps the
//                  rest: ua_machine_load / ua_machine_save)
//   disk-<n>.udisk the machine's disks, sparse udisk images
//   nvram.kv       UEFI variables and TPM state (native engine; the WebKit engine keeps them in
//                  memory until M4's follow-up)
//   artifacts/     a kernel and initramfs to boot directly ("kernel", "initrd"), when it has one
//   app.json       what only the app knows: the engine, the CD's bookmark. Not in machine.toml:
//                  the CLI drops keys it does not know when it saves.
//
// The CLI writes absolute disk paths; an iOS app's container moves when the app is reinstalled,
// so a disk path that does not exist is looked for by name in the machine's own folder.

// The CD a machine boots from: a file the user picked (a security-scoped bookmark: the file
// stays where it is) or one in the app's Documents (by name, for `-new-machine` and tests).
enum CDROM: Equatable {
    case bookmark(Data)
    case documents(String)
}

struct MachineDefinition {
    var memoryMib = 2048
    var cpus = 1
    // "uefi" boots through the bundled firmware; "none" enters the kernel directly. The CLI's
    // "bios" is booted as "uefi" here.
    var firmware = "uefi"
    var display = DisplayChoice.terminal
    var serialConsole = true
    var tpm = false
    var secureBoot = false
    var cmdlineAppend = ""
    var disks: [DiskImage] = []
    // A NIC: the CLI's `[[networks]]`, kept as they were (the app only knows on or off).
    var networks: [[String: Any]] = []
    var network: Bool {
        get { !networks.isEmpty }
        set {
            if !newValue { networks = [] } else if networks.isEmpty { networks = [MachineStore.userNetwork] }
        }
    }
    // app.json
    var engine: EngineKind?
    var bootsKernel = false
    var cdrom: CDROM?
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
    private static let errorCapacity = 1024
    // The CLI's `--display` values for the app's choices.
    private static let displayNames: [DisplayChoice: String] = [.terminal: "serial", .screen: "framebuffer", .both: "both"]
    // What the CLI's machine names allow (machines.rs `validate_name`).
    private static let maxNameLength = 64
    // What `uemulator quick create` writes for its NAT NIC.
    nonisolated static let userNetwork: [String: Any] = ["model": "virtio", "link": "user"]

    init() {
        root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Machines")
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
    func kernel(_ name: String) -> URL { directory(name).appendingPathComponent(Self.artifacts).appendingPathComponent("kernel") }
    func initrd(_ name: String) -> URL { directory(name).appendingPathComponent(Self.artifacts).appendingPathComponent("initrd") }

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
        else { return "Use up to \(maxNameLength) letters, digits, '.', '_' or '-', starting with a letter or digit" }
        return nil
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
        if let firmware = table["firmware"] as? String { definition.firmware = firmware }
        if let display = table["display"] as? String,
           let choice = Self.displayNames.first(where: { $0.value == display })?.key {
            definition.display = choice
        }
        // The CLI's default is one virtio console.
        if let consoles = table["consoles"] as? [String] { definition.serialConsole = consoles.first == "serial" }
        definition.tpm = table["tpm"] as? Bool ?? false
        definition.secureBoot = table["secure_boot"] as? Bool ?? false
        definition.cmdlineAppend = table["append"] as? String ?? ""
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
        let changes: [String: Any] = [
            "memory": definition.memoryMib,
            "cpus": definition.cpus,
            "firmware": definition.firmware,
            "display": Self.displayNames[definition.display] ?? "serial",
            "consoles": [definition.serialConsole ? "serial" : "virtio"],
            "tpm": definition.tpm,
            "secure_boot": definition.secureBoot,
            "append": definition.cmdlineAppend.isEmpty ? NSNull() : definition.cmdlineAppend,
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
        switch definition.cdrom {
        case .bookmark(let data): sidecar.cdromBookmark = data
        case .documents(let file): sidecar.cdromDocumentsFile = file
        case nil: break
        }
        try JSONEncoder().encode(sidecar).write(to: folder.appendingPathComponent(Self.sidecarFile), options: .atomic)
    }

    enum Source {
        // The bundled Alpine demo (hack/web/demo.py's kernel and initramfs), booted directly.
        case demo
        // A CD image, by bookmark or in Documents.
        case cdrom(CDROM)
        // Nothing yet: insert a CD later, or a disk that already has a system.
        case empty
    }

    func create(_ name: String, source: Source, diskMib: UInt64, memoryMib: Int, display: DisplayChoice, network: Bool = true) throws {
        if let problem = Self.validate(name) { throw MachineStoreError.failed(problem) }
        let folder = directory(name)
        guard !FileManager.default.fileExists(atPath: folder.path) else {
            throw MachineStoreError.failed("A machine named \(name) already exists")
        }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        do {
            var definition = MachineDefinition(memoryMib: memoryMib, display: display)
            definition.network = network
            if diskMib > 0 {
                // The CLI's name for a machine's first disk.
                let disk = folder.appendingPathComponent("disk-0.udisk")
                var error = [CChar](repeating: 0, count: Self.errorCapacity)
                guard ua_disk_create(disk.path, diskMib, &error, error.count) == 0 else {
                    throw MachineStoreError.failed(String(cString: error))
                }
                definition.disks = [DiskImage(url: disk, readOnly: false)]
            }
            switch source {
            case .demo:
                let resources = Bundle.main.resourceURL!.appendingPathComponent("demo")
                try FileManager.default.createDirectory(at: kernel(name).deletingLastPathComponent(), withIntermediateDirectories: true)
                try FileManager.default.copyItem(at: resources.appendingPathComponent("vmlinuz-virt"), to: kernel(name))
                try FileManager.default.copyItem(at: resources.appendingPathComponent("initramfs.gz"), to: initrd(name))
            case .cdrom(let cdrom):
                definition.cdrom = cdrom
            case .empty:
                break
            }
            try save(definition, as: name)
        } catch {
            try? FileManager.default.removeItem(at: folder)
            throw error
        }
        reload()
    }

    func delete(_ name: String) throws {
        models.removeValue(forKey: name)?.stop()
        try FileManager.default.removeItem(at: directory(name))
        reload()
    }

    // A bookmark for a file the user picked, to find it again next launch.
    static func bookmark(for url: URL) throws -> Data {
        #if os(macOS)
        try url.bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
        #else
        try url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
        #endif
    }

    static func resolve(_ cdrom: CDROM) -> URL? {
        switch cdrom {
        case .documents(let file):
            return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.appendingPathComponent(file)
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
        var cdromBookmark: Data?
        var cdromDocumentsFile: String?
    }
}
