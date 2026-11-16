import SwiftTerm
import SwiftUI

// The guest's console in SwiftTerm's terminal view (UIView on iOS, NSView on macOS). The view is
// owned by the session so it keeps its scrollback across SwiftUI updates.
@MainActor
final class TerminalSession: NSObject, TerminalViewDelegate {
    let view = TerminalView(frame: CGRect(x: 0, y: 0, width: 640, height: 400))
    var onInput: (([UInt8]) -> Void)?
    var onResize: ((Int, Int) -> Void)?
    // The last size the view reported; SwiftTerm has no public getter for it. 80x25 until the
    // first layout, the guest console's own default.
    private(set) var size = (cols: 80, rows: 25)
    // Points; a phone's narrower screen gets the smaller size so 80 columns fit across in landscape.
    #if os(macOS)
    private static let fontSize: CGFloat = 12
    #else
    private static let fontSize: CGFloat = 11
    #endif

    override init() {
        super.init()
        view.terminalDelegate = self
        #if os(macOS)
        view.font = NSFont.monospacedSystemFont(ofSize: Self.fontSize, weight: .regular)
        #else
        view.font = UIFont.monospacedSystemFont(ofSize: Self.fontSize, weight: .regular)
        // After TerminalView's own setup, which installs SwiftTerm's bar once, in init.
        view.inputAccessoryView = accessory
        accessory.onKey = { [weak self] usage, pressed in self?.accessoryKey(usage, pressed: pressed) }
        accessory.onCharacter = { [weak self] character in self?.view.insertText(String(character)) }
        // TerminalView applies its own `controlModifier` to the next character typed and clears it.
        accessory.onControlLatched = { [weak self] latched in self?.view.controlModifier = latched }
        accessory.onHide = { [weak self] in _ = self?.view.resignFirstResponder() }
        #endif
    }

    func feed(_ bytes: [UInt8]) {
        view.feed(byteArray: bytes[...])
    }

    func reset() {
        view.resetToInitialState()
    }

    nonisolated func send(source: TerminalView, data: ArraySlice<UInt8>) {
        let bytes = Array(data)
        MainActor.assumeIsolated {
            onInput?(bytes)
            #if os(iOS)
            // A character from the system keyboard used up the latched ctrl; unlight the bar's key.
            // TerminalView clears its flag after this call returns, hence the next turn.
            DispatchQueue.main.async { [weak self] in
                guard let self, self.accessory.controlLatched, !self.view.controlModifier else { return }
                self.accessory.controlLatched = false
            }
            #endif
        }
    }

    nonisolated func sizeChanged(source: TerminalView, newCols: Int, newRows: Int) {
        MainActor.assumeIsolated {
            size = (newCols, newRows)
            onResize?(newCols, newRows)
        }
    }

    #if os(iOS)
    private let accessory = KeyAccessory()
    // A held arrow on the bar: terminal bytes have no key up, so the repeat is ours.
    private var repeatTimer: Timer?
    private static let repeatDelay: TimeInterval = 0.5
    private static let repeatInterval: TimeInterval = 0.1

    private func accessoryKey(_ usage: UInt16, pressed: Bool) {
        repeatTimer?.invalidate()
        repeatTimer = nil
        guard pressed else { return }
        switch usage {
        case HIDUsage.escape: view.send([Self.escapeByte])
        case HIDUsage.tab: view.send([Self.tabByte])
        case HIDUsage.up, HIDUsage.down, HIDUsage.left, HIDUsage.right:
            // TerminalView's own senders follow the guest's cursor-key mode.
            let sendArrow: () -> Void = switch usage {
            case HIDUsage.up: { [weak self] in self?.view.sendKeyUp() }
            case HIDUsage.down: { [weak self] in self?.view.sendKeyDown() }
            case HIDUsage.left: { [weak self] in self?.view.sendKeyLeft() }
            default: { [weak self] in self?.view.sendKeyRight() }
            }
            sendArrow()
            let timer = Timer(timeInterval: Self.repeatDelay, repeats: false) { [weak self] _ in
                MainActor.assumeIsolated {
                    guard let self else { return }
                    let repeating = Timer(timeInterval: Self.repeatInterval, repeats: true) { _ in
                        MainActor.assumeIsolated { sendArrow() }
                    }
                    RunLoop.main.add(repeating, forMode: .common)
                    self.repeatTimer = repeating
                }
            }
            RunLoop.main.add(timer, forMode: .common)
            repeatTimer = timer
        case let key where key >= HIDUsage.f1 && Int(key - HIDUsage.f1) < EscapeSequences.cmdF.count:
            view.send(EscapeSequences.cmdF[Int(key - HIDUsage.f1)])
        default: break
        }
    }

    private static let escapeByte: UInt8 = 0x1B
    private static let tabByte: UInt8 = 0x09
    #endif

    nonisolated func setTerminalTitle(source: TerminalView, title: String) {}
    nonisolated func hostCurrentDirectoryUpdate(source: TerminalView, directory: String?) {}
    nonisolated func scrolled(source: TerminalView, position: Double) {}
    nonisolated func requestOpenLink(source: TerminalView, link: String, params: [String: String]) {}
    nonisolated func clipboardCopy(source: TerminalView, content: Data) {}
    nonisolated func rangeChanged(source: TerminalView, startY: Int, endY: Int) {}
}

#if os(macOS)
struct TerminalPane: NSViewRepresentable {
    let session: TerminalSession
    func makeNSView(context: Context) -> TerminalView { session.view }
    func updateNSView(_ view: TerminalView, context: Context) {}
}
#else
struct TerminalPane: UIViewRepresentable {
    let session: TerminalSession
    func makeUIView(context: Context) -> TerminalView { session.view }
    func updateUIView(_ view: TerminalView, context: Context) {}
}
#endif
