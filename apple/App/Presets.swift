import Foundation

// The installers `uemulator quick create --install` sets up. crates/uemulator-cli/src/quick.rs is
// the source: keep the URLs, file names, command lines and memory here in step with it, and the
// raw values with its `machine.toml` spelling, so the CLI can boot an installer machine the app made.
enum Distro: String, CaseIterable, Identifiable {
    case alpine
    case debian
    case fedora

    var id: String { rawValue }

    var title: String {
        switch self {
        case .alpine: "Alpine Linux"
        case .debian: "Debian"
        case .fedora: "Fedora"
        }
    }

    // What `ua_bootstrap_initrd` takes for it. The same spelling as the raw value today, but the
    // raw value is machine.toml's and this is the C ABI's: they need not stay equal.
    var bootstrapName: String {
        switch self {
        case .alpine: "alpine"
        case .debian: "debian"
        case .fedora: "fedora"
        }
    }

    struct Download {
        // The name under the machine's artifacts/, as the CLI saves it.
        let file: String
        let url: URL
    }

    var kernel: Download {
        switch self {
        case .alpine: Download(file: "vmlinuz-virt", url: Self.alpineNetboot.appendingPathComponent("vmlinuz-virt"))
        case .debian: Download(file: "linux", url: Self.debianNetboot.appendingPathComponent("linux"))
        case .fedora: Download(file: "vmlinuz", url: Self.fedoraPxeboot.appendingPathComponent("vmlinuz"))
        }
    }

    var initrd: Download {
        switch self {
        case .alpine: Download(file: "initramfs-virt", url: Self.alpineNetboot.appendingPathComponent("initramfs-virt"))
        case .debian: Download(file: "initrd.gz", url: Self.debianNetboot.appendingPathComponent("initrd.gz"))
        case .fedora: Download(file: "initrd.img", url: Self.fedoraPxeboot.appendingPathComponent("initrd.img"))
        }
    }

    // Appended to the command line while the installer runs (`Distro::cmdline`).
    var cmdline: String {
        switch self {
        case .alpine:
            "earlycon=uart8250,io,0x3f8 ip=dhcp alpine_repo=https://dl-cdn.alpinelinux.org/alpine/latest-stable/main modloop=https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot/modloop-virt"
        case .debian:
            ""
        case .fedora:
            "ip=dhcp rd.neednet=1 inst.text inst.repo=https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Server/x86_64/os/"
        }
    }

    // Fedora's ~900 MB stage2 is downloaded into RAM beside Anaconda (`FEDORA_INSTALL_MEMORY_MIB`).
    var memoryMib: Int {
        self == .fedora ? 3072 : Self.easyMemoryMib
    }

    static let easyMemoryMib = 2048
    static let easyDiskGib = 64

    private static let alpineNetboot = URL(string: "https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot")!
    private static let debianNetboot = URL(string: "https://deb.debian.org/debian/dists/trixie/main/installer-amd64/current/images/netboot/debian-installer/amd64")!
    private static let fedoraPxeboot = URL(string: "https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Server/x86_64/os/images/pxeboot")!
}

// The bundled demos: hack/web/demo.py's Alpine kernel and initramfs, and Edera's Xen with that
// Alpine as dom0. Xen's arguments and IOMMU come from the bundled demo.json, so the app boots it
// as the web page does.
enum Demo: String, CaseIterable, Identifiable {
    case alpine
    case xen

    var id: String { rawValue }

    var title: String {
        switch self {
        case .alpine: "Alpine Linux"
        case .xen: "Edera Xen"
        }
    }

    // The machine name a new one gets (with -2, -3, ... when taken).
    var machineName: String {
        switch self {
        case .alpine: "alpine-demo"
        case .xen: "xen-demo"
        }
    }

    static let memoryMib = 1024
    // Tells the demo's init it runs in the app, for its welcome text (hack/web/demo.py).
    static let hostArgument = "uemulator.host=app"

    static var resources: URL { Bundle.main.resourceURL!.appendingPathComponent("demo") }

    struct Manifest: Decodable {
        struct Machine: Decodable {
            let id: String
            let hypervisor: String?
            let hypervisorArgs: String?
            let iommu: Bool?
        }
        let machines: [Machine]
    }

    var manifest: Manifest.Machine? {
        guard let data = try? Data(contentsOf: Self.resources.appendingPathComponent("demo.json")),
              let manifest = try? JSONDecoder().decode(Manifest.self, from: data)
        else { return nil }
        return manifest.machines.first { $0.id == rawValue }
    }
}

enum DownloadError: LocalizedError {
    case status(URL, Int)
    var errorDescription: String? {
        switch self {
        case .status(let url, let code): "\(url.lastPathComponent): HTTP \(code)"
        }
    }
}

// Fetches a file into place, reporting the fraction done. The file appears under its final name
// only once complete, so a cut download is fetched again rather than booted.
final class Downloader: NSObject, @unchecked Sendable {
    private var task: URLSessionDownloadTask?
    private var observation: NSKeyValueObservation?

    func fetch(_ url: URL, to destination: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                let task = URLSession.shared.downloadTask(with: url) { location, response, error in
                    if let error { return continuation.resume(throwing: error) }
                    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                        return continuation.resume(throwing: DownloadError.status(url, http.statusCode))
                    }
                    guard let location else { return continuation.resume(throwing: URLError(.badServerResponse)) }
                    do {
                        try? FileManager.default.removeItem(at: destination)
                        try FileManager.default.moveItem(at: location, to: destination)
                        continuation.resume()
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
                observation = task.progress.observe(\.fractionCompleted) { value, _ in progress(value.fractionCompleted) }
                self.task = task
                task.resume()
            }
        } onCancel: {
            task?.cancel()
        }
    }
}

// Copies a file the user picked into a machine's folder, off the main thread: an ISO from Files
// can be gigabytes. Inside the file's security scope; a name already taken gets " 2", " 3", ...
enum Importer {
    static func copy(_ source: URL, into folder: URL, as name: String? = nil) async throws -> URL {
        try await Task.detached(priority: .userInitiated) {
            let scoped = source.startAccessingSecurityScopedResource()
            defer { if scoped { source.stopAccessingSecurityScopedResource() } }
            let destination = name.map { folder.appendingPathComponent($0) } ?? uniqueURL(for: source.lastPathComponent, in: folder)
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.copyItem(at: source, to: destination)
            return destination
        }.value
    }

    static func uniqueURL(for file: String, in folder: URL) -> URL {
        let base = (file as NSString).deletingPathExtension
        let ext = (file as NSString).pathExtension
        var candidate = folder.appendingPathComponent(file)
        var number = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = folder.appendingPathComponent(ext.isEmpty ? "\(base) \(number)" : "\(base) \(number).\(ext)")
            number += 1
        }
        return candidate
    }
}
