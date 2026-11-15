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
final class NativeEngine: MachineEngine, @unchecked Sendable {
    // Called on the main actor.
    var onOutput: (([UInt8]) -> Void)?
    var onState: ((EngineState) -> Void)?
    // The library's log hook is not wired through the C ABI yet.
    var onLog: ((String) -> Void)?
    // Main actor only.
    var screen: ScreenSurface?

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
        let network: NetworkLink?
        func run() {
            // The session stops calling into the machine first; ua_destroy stops the client's
            // thread, after which nothing calls the link's sender.
            network?.detach()
            ua_destroy(machine)
            network?.release()
            Unmanaged<Run>.fromOpaque(context).release()
        }
    }

    // The machine's network: its Wisp client (Rust, `ua_network_start`) talks to a WispSession here,
    // in-process. The session answers on its own queue, possibly after the machine is gone, so it
    // reaches the machine only through `deliver`, which `detach` turns off under the lock.
    private final class NetworkLink: @unchecked Sendable {
        private let lock = NSLock()
        private var machine: OpaquePointer?
        private(set) var session: WispSession!
        // Retained for the C side from ua_network_start until ua_destroy has returned.
        private var retained: Unmanaged<NetworkLink>?

        init(machine: OpaquePointer) {
            self.machine = machine
            session = WispSession { [weak self] packet in self?.deliver(packet) }
        }

        func start() -> Bool {
            guard let machine else { return false }
            let context = Unmanaged.passRetained(self)
            retained = context
            guard ua_network_start(machine, NetworkLink.sendCallback, context.toOpaque()) == Int32(UA_RESULT_DONE) else {
                return false
            }
            ua_wisp_connecting(machine)
            session.start()
            return true
        }

        private func deliver(_ packet: Data) {
            lock.lock()
            defer { lock.unlock() }
            guard let machine else { return }
            _ = packet.withUnsafeBytes { ua_wisp_message(machine, $0.bindMemory(to: UInt8.self).baseAddress, $0.count) }
        }

        func detach() {
            lock.lock()
            machine = nil
            lock.unlock()
            session.close()
        }

        func release() {
            retained?.release()
            retained = nil
        }

        // The client's packets, on the network's thread: copied, then handed to the session.
        private static let sendCallback: ua_wisp_send_fn = { context, packet, len in
            guard let context, let packet else { return }
            let link = Unmanaged<NetworkLink>.fromOpaque(context).takeUnretainedValue()
            link.session.receive(Data(bytes: packet, count: len))
        }
    }

    private var network: NetworkLink?

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
    // How often the screen and the pointer image are looked at: a display refresh. Cheap when
    // nothing changed (a generation compare under the display's lock).
    private static let screenInterval: TimeInterval = 1.0 / 60
    // The library's pixel format, XRGB8888.
    private static let bytesPerPixel = 4
    private var screenTimer: Timer?
    private static let errorCapacity = 1024

    deinit {
        if let machine { ua_destroy(machine) }
        if let context { Unmanaged<Run>.fromOpaque(context).release() }
    }

    @MainActor
    func start(_ settings: MachineSettings) {
        stop()
        onState?(.starting)
        nextRunId += 1
        let runId = nextRunId
        let context = Unmanaged.passRetained(Run(engine: self, id: runId)).toOpaque()
        var error = [CChar](repeating: 0, count: NativeEngine.errorCapacity)
        var options = settings.options
        if settings.network { options |= UInt32(UA_OPTION_NETWORK) }
        // C strings for the disks' paths, alive until ua_create has copied them.
        let paths = settings.disks.map { strdup($0.url.path) }
        let nvram = settings.nvram.map { strdup($0.path) } ?? nil
        defer {
            paths.forEach { free($0) }
            free(nvram)
        }
        let disks = zip(settings.disks, paths).map { ua_disk(path: $1, read_only: $0.readOnly ? 1 : 0) }
        let created: OpaquePointer? = disks.withUnsafeBufferPointer { disks in settings.kernel.withUnsafeBytes { kernel in
            settings.initrd.withUnsafeBytes { initrd in
                (settings.firmware ?? Data()).withUnsafeBytes { firmware in
                    settings.cmdlineAppend.withCString { cmdline in
                        var config = ua_config(
                            kernel: kernel.bindMemory(to: UInt8.self).baseAddress, kernel_len: kernel.count,
                            initrd: initrd.bindMemory(to: UInt8.self).baseAddress, initrd_len: initrd.count,
                            firmware: firmware.bindMemory(to: UInt8.self).baseAddress, firmware_len: firmware.count,
                            cmdline_append: cmdline,
                            disks: disks.baseAddress, disk_count: disks.count,
                            nvram_path: nvram,
                            memory_mib: settings.memoryMib, cpus: settings.cpus, options: options,
                            cols: settings.cols, rows: settings.rows)
                        return ua_create(&config, NativeEngine.outputCallback, NativeEngine.exitCallback, context,
                                         &error, error.count)
                    }
                }
            }
        } }
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
        if settings.network {
            let link = NetworkLink(machine: created)
            if link.start() { network = link }
        }
        ua_start(created)
        if ua_has_display(created) != 0 {
            screen?.reset()
            let timer = Timer(timeInterval: Self.screenInterval, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated { self?.refreshScreen() }
            }
            RunLoop.main.add(timer, forMode: .common)
            screenTimer = timer
        }
        // One word for a phone's toolbar, beside the engine menu.
        onState?(.running(detail: jit ? "JIT" : "interp"))
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
        screenTimer?.invalidate()
        screenTimer = nil
        ua_stop(machine)
        let teardown = Teardown(machine: machine, context: context, network: network)
        network = nil
        DispatchQueue.global(qos: .userInitiated).async { teardown.run() }
        onState?(.stopped("Stopped"))
    }

    @MainActor
    func send(_ bytes: [UInt8]) {
        guard let machine else { return }
        bytes.withUnsafeBufferPointer { ua_input(machine, $0.baseAddress, $0.count) }
    }

    @MainActor
    func requestSize(width: Int, height: Int) {
        guard let machine, width > 0, height > 0 else { return }
        ua_request_size(machine, UInt32(width), UInt32(height))
    }

    // ScreenInput. The library queues the events for the vCPU; a key the guest's driver is not
    // listening for yet is dropped there.
    @MainActor
    func key(usage: UInt16, pressed: Bool) {
        guard let machine else { return }
        ua_key(machine, usage, pressed ? 1 : 0)
    }

    @MainActor
    func pointerMove(x: Int, y: Int) {
        guard let machine, x >= 0, y >= 0 else { return }
        ua_pointer_move(machine, UInt32(x), UInt32(y))
    }

    @MainActor
    func pointerButton(_ button: Int, pressed: Bool) {
        guard let machine, button >= 0 else { return }
        ua_pointer_button(machine, UInt32(button), pressed ? 1 : 0)
    }

    @MainActor
    func scroll(clicks: Int) {
        guard let machine else { return }
        ua_scroll(machine, Int32(clamping: clicks))
    }

    @MainActor
    func releaseAll() {
        guard let machine else { return }
        ua_release_all(machine)
    }

    // The library hands over the whole screen and the rectangles that differ from what the
    // surface already has; only those are copied.
    @MainActor
    private func refreshScreen() {
        guard let machine, let screen else { return }
        let context = Unmanaged.passUnretained(screen).toOpaque()
        ua_screen_update(machine, NativeEngine.screenCallback, context)
        ua_cursor_update(machine, NativeEngine.cursorCallback, context)
    }

    // Both run inside ua_*_update on the main thread, so the surface may be touched directly.
    private static let screenCallback: ua_screen_fn = { context, width, height, pixels, stride, rects, count in
        guard let context, let pixels else { return }
        let surface = Unmanaged<ScreenSurface>.fromOpaque(context).takeUnretainedValue()
        MainActor.assumeIsolated {
            surface.resize(width: Int(width), height: Int(height))
            for index in 0..<count {
                guard let rect = rects?[index] else { continue }
                let origin = UnsafeRawPointer(pixels) + Int(rect.y) * stride + Int(rect.x) * NativeEngine.bytesPerPixel
                surface.copy(ScreenSurface.Rect(x: Int(rect.x), y: Int(rect.y), width: Int(rect.width), height: Int(rect.height)),
                             from: origin, sourceStride: stride)
            }
            surface.commit()
        }
    }

    private static let cursorCallback: ua_cursor_fn = { context, kind, width, height, hotX, hotY, pixels, len in
        guard let context else { return }
        let surface = Unmanaged<ScreenSurface>.fromOpaque(context).takeUnretainedValue()
        let image = pixels.map { Data(bytes: $0, count: len) }
        MainActor.assumeIsolated {
            switch kind {
            case Int32(UA_CURSOR_HIDDEN): surface.setPointerHidden()
            case Int32(UA_CURSOR_IMAGE):
                surface.setPointer(width: Int(width), height: Int(height), hotX: Int(hotX), hotY: Int(hotY),
                                   pixels: image ?? Data())
            default: surface.setPointerHost()
            }
        }
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
