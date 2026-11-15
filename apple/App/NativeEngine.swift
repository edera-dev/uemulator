import Foundation
import UemulatorCore

// The machine in this process (docs/plans/28-apple.md, "NativeEngine"): crates/uemulator-apple
// through its C ABI. The JIT on macOS, the interpreter on iOS.
//
// Threads: the library calls back on its vCPU thread (console output, with the console locked)
// and on its runner thread (the exit). Both hand off to the main actor. Output is batched (one
// flush per `outputInterval`, as the web worker batches) because a boot prints in single bytes.
//
// Every machine gets a `Run` of its own as its callback context, and only the current run's
// callbacks reach the UI. That is what lets `stop` return at once: the old machine is torn down
// on a background queue (an interpreted vCPU can take a while to notice the stop, and
// `ua_destroy` waits for it), and whatever it still prints or reports meanwhile is dropped
// instead of landing in the next machine's terminal.
final class NativeEngine: @unchecked Sendable {
    enum State: Equatable {
        case starting
        case running(jit: Bool)
        case stopped(String)
    }

    struct Settings {
        var kernel: Data
        var initrd: Data
        var firmware: Data?
        var cmdlineAppend = ""
        var memoryMib: UInt32 = 256
        var cpus: UInt32 = 1
        var serialConsole = true
        var jit = true
        var cols: UInt16 = 80
        var rows: UInt16 = 25
    }

    // Called on the main actor.
    var onOutput: (([UInt8]) -> Void)?
    var onState: ((State) -> Void)?

    // One machine's callback context. Retained for the C side from `ua_create` until `ua_destroy`
    // has returned, so a callback never sees it freed.
    private final class Run: @unchecked Sendable {
        weak var engine: NativeEngine?
        let id: Int
        init(engine: NativeEngine, id: Int) {
            self.engine = engine
            self.id = id
        }
    }

    // A stopped machine and its context, handed to a background queue. Nothing else holds either
    // once `stop` has detached them, so sending the raw pointers across is safe.
    private struct Teardown: @unchecked Sendable {
        let machine: OpaquePointer
        let context: UnsafeMutableRawPointer
        func run() {
            ua_destroy(machine)
            Unmanaged<Run>.fromOpaque(context).release()
        }
    }

    // Main actor only.
    private var machine: OpaquePointer?
    private var context: UnsafeMutableRawPointer?
    private var nextRunId = 0

    // Guarded by `lock`: shared with the vCPU and runner threads.
    private let lock = NSLock()
    private var currentRunId: Int?
    private var pending: [UInt8] = []
    private var flushScheduled = false

    private static let outputInterval: DispatchTimeInterval = .milliseconds(16)
    private static let errorCapacity = 1024

    deinit {
        if let machine { ua_destroy(machine) }
        if let context { Unmanaged<Run>.fromOpaque(context).release() }
    }

    @MainActor
    func start(_ settings: Settings) {
        stop()
        onState?(.starting)
        nextRunId += 1
        let runId = nextRunId
        let context = Unmanaged.passRetained(Run(engine: self, id: runId)).toOpaque()
        var error = [CChar](repeating: 0, count: NativeEngine.errorCapacity)
        var options = UInt32(UA_OPTION_ACPI) | UInt32(UA_OPTION_RNG)
        if settings.jit { options |= UInt32(UA_OPTION_JIT) }
        if settings.serialConsole { options |= UInt32(UA_OPTION_SERIAL_CONSOLE) }
        let created: OpaquePointer? = settings.kernel.withUnsafeBytes { kernel in
            settings.initrd.withUnsafeBytes { initrd in
                (settings.firmware ?? Data()).withUnsafeBytes { firmware in
                    settings.cmdlineAppend.withCString { cmdline in
                        var config = ua_config(
                            kernel: kernel.bindMemory(to: UInt8.self).baseAddress, kernel_len: kernel.count,
                            initrd: initrd.bindMemory(to: UInt8.self).baseAddress, initrd_len: initrd.count,
                            firmware: firmware.bindMemory(to: UInt8.self).baseAddress, firmware_len: firmware.count,
                            cmdline_append: cmdline,
                            memory_mib: settings.memoryMib, cpus: settings.cpus, options: options,
                            cols: settings.cols, rows: settings.rows)
                        return ua_create(&config, NativeEngine.outputCallback, NativeEngine.exitCallback, context,
                                         &error, error.count)
                    }
                }
            }
        }
        guard let created else {
            Unmanaged<Run>.fromOpaque(context).release()
            onState?(.stopped(String(cString: error)))
            return
        }
        machine = created
        self.context = context
        lock.lock()
        currentRunId = runId
        lock.unlock()
        let jit = ua_jit_active(created) != 0
        ua_start(created)
        onState?(.running(jit: jit))
    }

    // Detaches the machine from the UI at once and tears it down in the background.
    @MainActor
    func stop() {
        guard let machine, let context else { return }
        self.machine = nil
        self.context = nil
        lock.lock()
        currentRunId = nil
        pending = []
        lock.unlock()
        ua_stop(machine)
        let teardown = Teardown(machine: machine, context: context)
        DispatchQueue.global(qos: .userInitiated).async { teardown.run() }
        onState?(.stopped("Stopped"))
    }

    @MainActor
    func send(_ bytes: [UInt8]) {
        guard let machine else { return }
        bytes.withUnsafeBufferPointer { ua_input(machine, $0.baseAddress, $0.count) }
    }

    @MainActor
    func resize(cols: Int, rows: Int) {
        guard let machine, cols > 0, rows > 0 else { return }
        ua_resize(machine, UInt16(clamping: cols), UInt16(clamping: rows))
    }

    private static let outputCallback: ua_output_fn = { context, bytes, len in
        guard let context, let bytes else { return }
        let run = Unmanaged<Run>.fromOpaque(context).takeUnretainedValue()
        run.engine?.receive(run: run.id, UnsafeBufferPointer(start: bytes, count: len))
    }

    private static let exitCallback: ua_exit_fn = { context, reason, message in
        guard let context else { return }
        let run = Unmanaged<Run>.fromOpaque(context).takeUnretainedValue()
        let text = message.map { String(cString: $0) } ?? ""
        run.engine?.exited(run: run.id, reason: reason, message: text)
    }

    private func isCurrent(_ run: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return currentRunId == run
    }

    // vCPU thread, console locked: copy and return.
    private func receive(run: Int, _ bytes: UnsafeBufferPointer<UInt8>) {
        lock.lock()
        guard currentRunId == run else {
            lock.unlock()
            return
        }
        pending.append(contentsOf: bytes)
        let schedule = !flushScheduled
        flushScheduled = true
        lock.unlock()
        if schedule {
            DispatchQueue.main.asyncAfter(deadline: .now() + NativeEngine.outputInterval) { [weak self] in self?.flush() }
        }
    }

    private func flush() {
        lock.lock()
        let bytes = pending
        pending = []
        flushScheduled = false
        lock.unlock()
        if !bytes.isEmpty { onOutput?(bytes) }
    }

    // Runner thread.
    private func exited(run: Int, reason: Int32, message: String) {
        let text: String
        switch reason {
        case Int32(UA_EXIT_SHUTDOWN): text = "Powered off"
        case Int32(UA_EXIT_RESET): text = "Reset"
        case Int32(UA_EXIT_TRIPLE_FAULT): text = "Triple fault"
        case Int32(UA_EXIT_STOPPED): text = "Stopped"
        default: text = "Emulator error: \(message)"
        }
        DispatchQueue.main.async { [weak self] in
            guard let self, self.isCurrent(run) else { return }
            self.flush()
            self.onState?(.stopped(text))
        }
    }
}
