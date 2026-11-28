import Security
import SwiftUI
import UemulatorCore
import UniformTypeIdentifiers

// Unattended installs, the app's `quick bootstrap` (docs/plans/35-quick-bootstrap.md, "App"): the
// Install sheet's answers go to the core over `ua_bootstrap_initrd`, which hashes the passwords
// and appends the distro's answer files to the installer's initramfs, so no template lives here.

// The CLI's `[bootstrap]` table, `BootstrapRecord` in crates/uemulator-cli/src/bootstrap.rs: keep
// the keys in step with it, or a machine the app made stops loading in the CLI (serde refuses a
// missing key) and Reinstall… loses the answers the CLI wrote. Never a password or its hash.
struct BootstrapRecord: Equatable {
    var user: String
    var hostname: String
    // The key lines themselves: the files they came from may be gone by the reinstall.
    var sshKeys: [String] = []
    var timezone: String
    var locale: String
    var keymap: String
    var packages: [String] = []
    // Whether root got a password; root is locked otherwise.
    var rootPassword = false

    // From machine.toml's table as `ua_machine_load` gives it; nil without the keys the CLI
    // requires.
    init?(table: [String: Any]) {
        guard let user = table["user"] as? String, let hostname = table["hostname"] as? String,
              let timezone = table["timezone"] as? String, let locale = table["locale"] as? String,
              let keymap = table["keymap"] as? String
        else { return nil }
        self.init(user: user, hostname: hostname, sshKeys: table["ssh_keys"] as? [String] ?? [],
                  timezone: timezone, locale: locale, keymap: keymap,
                  packages: table["packages"] as? [String] ?? [], rootPassword: table["root_password"] as? Bool ?? false)
    }

    init(user: String, hostname: String, sshKeys: [String], timezone: String, locale: String, keymap: String,
         packages: [String], rootPassword: Bool) {
        self.user = user
        self.hostname = hostname
        self.sshKeys = sshKeys
        self.timezone = timezone
        self.locale = locale
        self.keymap = keymap
        self.packages = packages
        self.rootPassword = rootPassword
    }

    var table: [String: Any] {
        ["user": user, "hostname": hostname, "ssh_keys": sshKeys, "timezone": timezone, "locale": locale,
         "keymap": keymap, "packages": packages, "root_password": rootPassword]
    }
}

// An unattended install waiting for its installer boot: the record plus the passwords, which live
// only in memory (`MachineStore.pendingBootstrap`). Nothing here may be printed or saved.
struct BootstrapRequest {
    var record: BootstrapRecord
    var password: String
    // Nil locks root.
    var rootPassword: String?

    // `ua_bootstrap_initrd`'s salt: 16 random bytes (the bootstrap crate's SALT_LEN).
    private static let saltBytes = 16
    private static let errorCapacity = 1024

    // `initrd` (the distro's installer initramfs) with the answer files appended, and the kernel
    // arguments that make the installer read them, to go after the distro's own.
    func extend(_ initrd: Data, distro: Distro) throws -> (initrd: Data, cmdline: String) {
        let answers: [String: Any] = [
            "user": record.user, "password": password, "root_password": rootPassword ?? NSNull(),
            "hostname": record.hostname, "ssh_keys": record.sshKeys, "timezone": record.timezone,
            "locale": record.locale, "keymap": record.keymap, "packages": record.packages,
        ]
        let json = String(decoding: try JSONSerialization.data(withJSONObject: answers), as: UTF8.self)
        var salt = [UInt8](repeating: 0, count: Self.saltBytes)
        guard SecRandomCopyBytes(kSecRandomDefault, salt.count, &salt) == errSecSuccess else {
            throw MachineStoreError.failed("No random bytes for the password's salt")
        }
        var out: UnsafeMutablePointer<UInt8>?
        var outLength = 0
        var arguments: UnsafeMutablePointer<CChar>?
        var error = [CChar](repeating: 0, count: Self.errorCapacity)
        let errorLength = error.count
        let result = initrd.withUnsafeBytes { base in
            ua_bootstrap_initrd(distro.bootstrapName, json, salt, salt.count,
                                base.bindMemory(to: UInt8.self).baseAddress, base.count,
                                &out, &outLength, &arguments, &error, errorLength)
        }
        guard result == 0, let out, let arguments else { throw MachineStoreError.failed(String(cString: error)) }
        let cmdline = String(cString: arguments)
        ua_string_free(arguments)
        // Handed over without a copy: an installer initramfs runs to a few hundred MB (Fedora's).
        let extended = Data(bytesNoCopy: out, count: outLength, deallocator: .custom { bytes, count in
            ua_bytes_free(bytes.assumingMemoryBound(to: UInt8.self), count)
        })
        return (extended, cmdline)
    }

    // What the core finds wrong with the answers (naming the field), or nil. Renders them against
    // an empty initramfs, so the sheet can say before the installer is even downloaded.
    func problem(for distro: Distro) -> String? {
        do {
            _ = try extend(Data(), distro: distro)
            return nil
        } catch {
            return error.localizedDescription
        }
    }
}

// Watches the console for UA_BOOTSTRAP_COMPLETION_MARKER, which the answer file prints once the
// install has succeeded, just before it reboots (the CLI's `MarkerWatch`). Only a reset after it
// boots the disk: a reset without it is an installer that failed or gave up.
struct CompletionMarkerWatch {
    private static let marker = Array(UA_BOOTSTRAP_COMPLETION_MARKER.utf8)
    // The last `marker.count - 1` bytes fed: enough to complete a marker split across chunks.
    private var tail: [UInt8] = []
    private(set) var seen = false

    mutating func feed(_ bytes: [UInt8]) {
        guard !seen else { return }
        let window = tail + bytes
        if window.firstRange(of: Self.marker) != nil {
            seen = true
            tail = []
        } else {
            tail = Array(window.suffix(Self.marker.count - 1))
        }
    }
}

// The Install sheet: the distro's unattended install (or, with "Install automatically" off, the
// interactive installer as before), and Reinstall…'s questions for a machine that has one.
struct BootstrapSheet: View {
    enum Purpose {
        // A new machine; the toggle chooses an unattended or interactive install.
        case create
        // An existing machine's disk is erased: always unattended, behind a destructive button.
        case reinstall
    }

    let distro: Distro
    let purpose: Purpose
    // Makes the machine (or reinstalls it) with the request, nil for an interactive install;
    // returns why it could not, which the sheet shows.
    let onSubmit: (BootstrapRequest?) -> String?
    @Environment(\.dismiss) private var dismiss
    @State private var automatic = true
    @State private var user: String
    @State private var password = ""
    @State private var rootPassword = ""
    @State private var hostname: String
    @State private var sshKeys: [String]
    @State private var sshKeyFile: String?
    @State private var timezone: String
    @State private var locale: String
    @State private var keymap: String
    @State private var packages: String
    @State private var showingMore = false
    @State private var pickingKey = false
    @State private var problem: String?

    // The CLI's fallbacks, where the device says nothing an installer can use.
    private static let fallbackLocale = "en_US.UTF-8"
    static let defaultKeymap = "us"
    private static let localeEncoding = ".UTF-8"

    // `hostname` is the default for a new machine (its name); `record` prefills a reinstall.
    init(distro: Distro, purpose: Purpose, hostname: String, record: BootstrapRecord?,
         onSubmit: @escaping (BootstrapRequest?) -> String?) {
        self.distro = distro
        self.purpose = purpose
        self.onSubmit = onSubmit
        _user = State(initialValue: record?.user ?? "")
        _hostname = State(initialValue: record?.hostname ?? hostname)
        _sshKeys = State(initialValue: record?.sshKeys ?? [])
        _sshKeyFile = State(initialValue: record.flatMap { $0.sshKeys.isEmpty ? nil : Self.keysTitle($0.sshKeys.count) })
        _timezone = State(initialValue: record?.timezone ?? TimeZone.current.identifier)
        _locale = State(initialValue: record?.locale ?? Self.deviceLocale)
        _keymap = State(initialValue: record?.keymap ?? Self.defaultKeymap)
        _packages = State(initialValue: record?.packages.joined(separator: " ") ?? "")
        _rootHadPassword = State(initialValue: purpose == .reinstall && record?.rootPassword == true)
    }

    // `de_DE.UTF-8` from the device's language and region, when that is a locale the installers
    // list: a two- or three-letter language and a two-letter country, as the CLI's
    // `locale_from_lang` takes them. Anything else (`es_419`, a region without a country) is the
    // fallback, or the sheet would open with answers the core refuses.
    static var deviceLocale: String {
        let languageLengths = 2...3
        let countryLength = 2
        guard let language = Locale.current.language.languageCode?.identifier,
              let region = Locale.current.region?.identifier,
              languageLengths.contains(language.count), language.allSatisfy({ $0.isASCII && $0.isLowercase }),
              region.count == countryLength, region.allSatisfy({ $0.isASCII && $0.isUppercase })
        else { return fallbackLocale }
        return "\(language)_\(region)\(localeEncoding)"
    }

    // The core's errors start with the field's name; these are the fields behind "More".
    private static let moreFields = ["timezone", "locale", "keymap", "package"]

    private static func keysTitle(_ count: Int) -> String {
        count == 1 ? "1 key" : "\(count) keys"
    }

    private var unattended: Bool { purpose == .reinstall || automatic }

    // A reinstall of a machine whose root had a password asks for it again: left empty, the
    // reinstalled root would quietly be locked.
    @State private var rootHadPassword = false
    // ... unless the user chooses to lock it this time.
    @State private var lockRoot = false
    private var rootPasswordRequired: Bool { rootHadPassword && !lockRoot }

    var body: some View {
        NavigationStack {
            Form {
                if purpose == .create {
                    Section {
                        Toggle("Install automatically", isOn: $automatic)
                    } footer: {
                        Text(automatic ? "The installer runs with these answers and no questions, then the machine starts on its disk."
                                       : "The installer asks its own questions on the terminal.")
                    }
                }
                if unattended {
                    answersSection
                    Section {
                        DisclosureGroup("More", isExpanded: $showingMore) {
                            TextField("Timezone", text: $timezone)
                            TextField("Locale", text: $locale)
                            TextField("Keymap", text: $keymap)
                            TextField("Packages", text: $packages, prompt: Text("space separated"))
                        }
                        .autocorrectionDisabled()
                        #if os(iOS)
                        .textInputAutocapitalization(.never)
                        #endif
                    }
                }
                if purpose == .reinstall {
                    Section {
                        Label("The machine's disk is erased and \(distro.title) installed again.", systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.orange)
                    }
                }
                if let problem {
                    Section {
                        Label(problem, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
                    }
                }
            }
            .formStyle(.grouped)
            .navigationTitle(purpose == .create ? "Install \(distro.title)" : "Reinstall \(distro.title)")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .fileImporter(isPresented: $pickingKey, allowedContentTypes: [.data, .text]) { result in
                guard case .success(let url) = result else { return }
                readKeys(url)
            }
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button(purpose == .create ? "Create" : "Erase and Install", role: purpose == .create ? nil : .destructive) {
                        submit()
                    }
                    .disabled(unattended && (user.isEmpty || password.isEmpty || (rootPasswordRequired && rootPassword.isEmpty)))
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 440, minHeight: 440)
        #else
        .formSheetSizing()
        #endif
    }

    private var answersSection: some View {
        Section {
            TextField("User", text: $user)
                .textContentType(.username)
            SecureField("Password", text: $password)
                .textContentType(.newPassword)
            if rootHadPassword {
                Toggle("Lock root", isOn: $lockRoot)
            }
            if !lockRoot {
                SecureField("Root password", text: $rootPassword, prompt: Text(rootPasswordRequired ? "required" : "optional"))
                    .textContentType(.newPassword)
            }
            TextField("Hostname", text: $hostname)
            LabeledContent("SSH key") {
                HStack {
                    Button(sshKeyFile ?? "Choose…") { pickingKey = true }
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if sshKeyFile != nil {
                        Button("Clear", systemImage: "xmark.circle.fill") {
                            sshKeys = []
                            sshKeyFile = nil
                        }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                    }
                }
            }
        } footer: {
            Text("Without a root password, root is locked and \(user.isEmpty ? "the user" : user) uses sudo. The passwords are not saved: Reinstall… asks again.")
        }
        .autocorrectionDisabled()
        #if os(iOS)
        .textInputAutocapitalization(.never)
        #endif
    }

    // A public key file or an authorized_keys file: one key per line, comments and blanks skipped.
    static func sshKeys(in text: String) -> [String] {
        text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
    }

    private func readKeys(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            problem = "\(url.lastPathComponent) is not a text file"
            return
        }
        let keys = Self.sshKeys(in: text)
        guard !keys.isEmpty else {
            problem = "\(url.lastPathComponent) has no SSH key in it"
            return
        }
        problem = nil
        sshKeys = keys
        sshKeyFile = url.lastPathComponent
    }

    private func submit() {
        var request: BootstrapRequest?
        if unattended {
            // A typed password the user then locked root over is not used.
            let root = lockRoot || rootPassword.isEmpty ? nil : rootPassword
            let record = BootstrapRecord(
                user: user, hostname: hostname, sshKeys: sshKeys, timezone: timezone, locale: locale, keymap: keymap,
                packages: packages.split(separator: " ").map(String.init), rootPassword: root != nil)
            let answers = BootstrapRequest(record: record, password: password, rootPassword: root)
            if let invalid = answers.problem(for: distro) {
                problem = invalid
                if Self.moreFields.contains(where: invalid.hasPrefix) { showingMore = true }
                return
            }
            request = answers
        }
        if let failure = onSubmit(request) {
            problem = failure
            return
        }
        dismiss()
    }
}

// The machine settings' rows for a recorded unattended install's answers (Reinstall… is the
// settings' own button).
struct BootstrapRecordRows: View {
    let record: BootstrapRecord

    var body: some View {
        LabeledContent("User", value: record.user)
        LabeledContent("Hostname", value: record.hostname)
        LabeledContent("Root", value: record.rootPassword ? "Password" : "Locked")
        if !record.sshKeys.isEmpty {
            LabeledContent("SSH Keys", value: "\(record.sshKeys.count)")
        }
        LabeledContent("Timezone", value: record.timezone)
        LabeledContent("Locale", value: record.locale)
        LabeledContent("Keymap", value: record.keymap)
        if !record.packages.isEmpty {
            LabeledContent("Packages", value: record.packages.joined(separator: " "))
        }
    }
}
