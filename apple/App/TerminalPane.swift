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
        for bar in [accessory, foldedKeyBar] {
            bar.onKey = { [weak self] usage, pressed in self?.accessoryKey(usage, pressed: pressed) }
            bar.onCharacter = { [weak self] character in self?.view.insertText(String(character)) }
            // TerminalView applies its own `controlModifier` to the next character typed and clears it.
            bar.onControlLatched = { [weak self] latched in self?.view.controlModifier = latched }
            bar.onHide = { [weak self] in _ = self?.view.resignFirstResponder() }
        }
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
                guard let self, self.keyBar.controlLatched, !self.view.controlModifier else { return }
                self.keyBar.controlLatched = false
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
    // Folded, the keys below the fold in FoldedControls instead of on the keyboard, where they would
    // cross it (ScreenUIView.folded says why).
    let foldedKeyBar = KeyAccessory(freestanding: true)
    var folded = false {
        didSet {
            guard folded != oldValue else { return }
            view.inputAccessoryView = folded ? nil : accessory
            if view.isFirstResponder { view.reloadInputViews() }
            container.followsKeyboard = !folded
        }
    }
    private var keyBar: KeyAccessory { folded ? foldedKeyBar : accessory }
    // What TerminalPane shows: the terminal, kept above the keyboard.
    private(set) lazy var container = TerminalContainer(terminal: view)
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
    func makeUIView(context: Context) -> TerminalContainer { session.container }
    func updateUIView(_ view: TerminalContainer, context: Context) {}
}

// Holds the terminal with its bottom on the keyboard's top (key bar included) while the terminal
// has the keyboard, so the rows shrink and the cursor line stays in sight; SwiftTerm itself
// leaves the keyboard over its last rows. Only the terminal's own keyboard counts: the screen's
// would otherwise resize the hidden console under it. Folded, the keyboard is below the fold and
// the terminal above it, so it keeps its full height.
final class TerminalContainer: UIView {
    private let terminal: TerminalView
    private var toBottom: NSLayoutConstraint!
    private var toKeyboard: NSLayoutConstraint!
    var followsKeyboard = true {
        didSet { if followsKeyboard != oldValue { update() } }
    }

    init(terminal: TerminalView) {
        self.terminal = terminal
        super.init(frame: terminal.frame)
        terminal.translatesAutoresizingMaskIntoConstraints = false
        addSubview(terminal)
        keyboardLayoutGuide.followsUndockedKeyboard = true
        // Hidden, the guide is this view's bottom rather than the safe area's: the terminal goes
        // under the home indicator as it did before, in full screen too.
        keyboardLayoutGuide.usesBottomSafeArea = false
        toBottom = terminal.bottomAnchor.constraint(equalTo: bottomAnchor)
        toKeyboard = terminal.bottomAnchor.constraint(equalTo: keyboardLayoutGuide.topAnchor)
        // Just short of required, so the `lessThanOrEqualTo` below wins when the two disagree.
        toKeyboard.priority = .required - 1
        NSLayoutConstraint.activate([
            terminal.topAnchor.constraint(equalTo: topAnchor),
            terminal.leadingAnchor.constraint(equalTo: leadingAnchor),
            terminal.trailingAnchor.constraint(equalTo: trailingAnchor),
            // A keyboard whose top is past this view's bottom never makes the terminal taller.
            terminal.bottomAnchor.constraint(lessThanOrEqualTo: bottomAnchor),
            toBottom,
        ])
        for name in [UIResponder.keyboardWillShowNotification, UIResponder.keyboardWillHideNotification,
                     UIResponder.keyboardWillChangeFrameNotification] {
            NotificationCenter.default.addObserver(self, selector: #selector(keyboardChanging), name: name, object: nil)
        }
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    // Sent before the keyboard animates, and after the first responder has changed, so the
    // terminal moves with the keyboard rather than after it.
    @objc private func keyboardChanging() { update() }

    private func update() {
        let follow = followsKeyboard && terminal.isFirstResponder
        guard follow != toKeyboard.isActive else { return }
        // Deactivate before activating: both active at once is a conflict.
        if follow {
            toBottom.isActive = false
            toKeyboard.isActive = true
        } else {
            toKeyboard.isActive = false
            toBottom.isActive = true
        }
    }
}
#endif
