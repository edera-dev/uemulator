import SwiftUI

// The guest's screen (docs/plans/24-display-input.md): the surface's image, aspect-fit, and the
// keyboard and pointer sent to the guest's virtio keyboard and tablet. Keys go as USB HID usages
// (positions, not characters: the guest applies its own layout), the pointer as absolute screen
// pixels, so the guest's pointer sits exactly under the host's and the guest's pointer image is
// shown as the host pointer rather than drawn into the frames.

#if os(macOS)
import AppKit

final class ScreenNSView: NSView {
    weak var input: (any ScreenInput)?
    var surface: ScreenSurface? {
        didSet { attach() }
    }
    private var trackingArea: NSTrackingArea?
    // Precise (trackpad) scrolling arrives in points; this many make one wheel notch.
    private static let pointsPerNotch: CGFloat = 12
    private var scrollRemainder: CGFloat = 0
    // NX_DEVICE*KEYMASK: which side's modifier is down, which NSEvent.ModifierFlags does not say.
    private static let deviceModifierMasks: [UInt16: UInt] = [
        56: 0x02, 60: 0x04, // left, right shift
        59: 0x01, 62: 0x2000, // left, right control
        58: 0x20, 61: 0x40, // left, right option
        55: 0x08, 54: 0x10, // left, right command
    ]
    private static let capsLockKeyCode: UInt16 = 57
    private static let invisibleCursor = NSCursor(image: NSImage(size: NSSize(width: 1, height: 1)), hotSpot: .zero)

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.backgroundColor = NSColor.black.cgColor
        layer?.contentsGravity = .resizeAspect
        // Pixel text magnified: nearest keeps it sharp; minified: linear keeps it legible.
        layer?.magnificationFilter = .nearest
        layer?.minificationFilter = .linear
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    override var acceptsFirstResponder: Bool { true }
    override var isFlipped: Bool { true }

    private func attach() {
        surface?.onImage = { [weak self] in self?.layer?.contents = self?.surface?.image }
        surface?.onPointer = { [weak self] in self?.window?.invalidateCursorRects(for: self!) }
        layer?.contents = surface?.image
    }

    override func resignFirstResponder() -> Bool {
        // Keys held while the focus leaves would never see their release.
        input?.releaseAll()
        return super.resignFirstResponder()
    }

    override func keyDown(with event: NSEvent) {
        // The guest repeats keys itself.
        guard !event.isARepeat, let usage = HIDUsage.forMacKeyCode[event.keyCode] else { return }
        input?.key(usage: usage, pressed: true)
    }

    override func keyUp(with event: NSEvent) {
        guard let usage = HIDUsage.forMacKeyCode[event.keyCode] else { return }
        input?.key(usage: usage, pressed: false)
    }

    override func flagsChanged(with event: NSEvent) {
        guard let usage = HIDUsage.forMacKeyCode[event.keyCode] else { return }
        if event.keyCode == Self.capsLockKeyCode {
            // macOS reports caps lock as a toggle; the guest wants the key itself.
            input?.key(usage: usage, pressed: true)
            input?.key(usage: usage, pressed: false)
            return
        }
        guard let mask = Self.deviceModifierMasks[event.keyCode] else { return }
        input?.key(usage: usage, pressed: event.modifierFlags.rawValue & mask != 0)
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let trackingArea { removeTrackingArea(trackingArea) }
        let area = NSTrackingArea(rect: .zero, options: [.mouseMoved, .activeInKeyWindow, .inVisibleRect, .cursorUpdate],
                                  owner: self, userInfo: nil)
        addTrackingArea(area)
        trackingArea = area
    }

    private func move(_ event: NSEvent) {
        guard let surface, surface.width > 0 else { return }
        let point = convert(event.locationInWindow, from: nil)
        let size = CGSize(width: surface.width, height: surface.height)
        guard let pixel = AspectFit.pixel(at: point, imageSize: size, in: bounds) else { return }
        input?.pointerMove(x: pixel.x, y: pixel.y)
    }

    override func mouseMoved(with event: NSEvent) { move(event) }
    override func mouseDragged(with event: NSEvent) { move(event) }
    override func rightMouseDragged(with event: NSEvent) { move(event) }
    override func otherMouseDragged(with event: NSEvent) { move(event) }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        move(event)
        input?.pointerButton(0, pressed: true)
    }

    override func mouseUp(with event: NSEvent) { input?.pointerButton(0, pressed: false) }
    override func rightMouseDown(with event: NSEvent) { move(event); input?.pointerButton(1, pressed: true) }
    override func rightMouseUp(with event: NSEvent) { input?.pointerButton(1, pressed: false) }

    // NSEvent numbers the other buttons 2 middle, 3 back, 4 forward, as the tablet does.
    override func otherMouseDown(with event: NSEvent) { move(event); input?.pointerButton(event.buttonNumber, pressed: true) }
    override func otherMouseUp(with event: NSEvent) { input?.pointerButton(event.buttonNumber, pressed: false) }

    override func scrollWheel(with event: NSEvent) {
        if event.hasPreciseScrollingDeltas {
            scrollRemainder += event.scrollingDeltaY
            let notches = Int(scrollRemainder / Self.pointsPerNotch)
            scrollRemainder -= CGFloat(notches) * Self.pointsPerNotch
            if notches != 0 { input?.scroll(clicks: notches) }
        } else if event.scrollingDeltaY != 0 {
            input?.scroll(clicks: event.scrollingDeltaY > 0 ? 1 : -1)
        }
    }

    override func resetCursorRects() {
        let cursor: NSCursor
        switch surface?.pointer ?? .host {
        case .host: cursor = .arrow
        case .hidden: cursor = Self.invisibleCursor
        case .image(let image, let hotX, let hotY):
            cursor = NSCursor(image: NSImage(cgImage: image, size: NSSize(width: image.width, height: image.height)),
                              hotSpot: NSPoint(x: hotX, y: hotY))
        }
        addCursorRect(bounds, cursor: cursor)
    }
}

struct ScreenPane: NSViewRepresentable {
    let surface: ScreenSurface
    let input: any ScreenInput

    func makeNSView(context: Context) -> ScreenNSView {
        let view = ScreenNSView(frame: .zero)
        view.surface = surface
        view.input = input
        DispatchQueue.main.async { view.window?.makeFirstResponder(view) }
        return view
    }

    func updateNSView(_ view: ScreenNSView, context: Context) {
        view.input = input
    }
}

#else
import UIKit

// The bar above the software keyboard, for the screen and the terminal alike: esc, a sticky ctrl,
// tab, a few symbols, F-keys, arrows, hide. It reports keys as HID usages; the screen sends them as
// keys and `TerminalSession` turns them into terminal bytes. A key goes down on touch and up on
// release, so a held arrow repeats through the guest's own repeat on the screen. The terminal uses
// this instead of SwiftTerm's own bar because that bar's keyboard key swaps the system keyboard
// for SwiftTerm's key grid rather than hiding it.
final class KeyAccessory: UIInputView {
    var onKey: ((UInt16, Bool) -> Void)?
    var onCharacter: ((Character) -> Void)?
    var onHide: (() -> Void)?
    // Applies to the next key, typed or from this bar, then lets go.
    var controlLatched = false {
        didSet {
            controlButton.isSelected = controlLatched
            if controlLatched != oldValue { onControlLatched?(controlLatched) }
        }
    }
    var onControlLatched: ((Bool) -> Void)?
    private var controlButton: UIButton!
    private static let barHeight: CGFloat = 44
    private static let initialWidth: CGFloat = 320
    private static let spacing: CGFloat = 4
    // Esc, ctrl, tab and the arrows at this width leave the symbols and F-keys room to scroll even
    // on a phone held upright.
    private static let fixedKeyWidth: CGFloat = 36
    private static let symbols: [Character] = ["~", "|", "/", "-"]
    private static let functionKeys = 12

    init() {
        // UIKit sets the width to the keyboard's; without the mask the bar keeps its initial one.
        super.init(frame: CGRect(x: 0, y: 0, width: Self.initialWidth, height: Self.barHeight), inputViewStyle: .keyboard)
        autoresizingMask = .flexibleWidth
        let leading = UIStackView(arrangedSubviews: [
            keyButton(symbol: "escape", usage: HIDUsage.escape),
            controlToggle(),
            keyButton(symbol: "arrow.right.to.line.compact", usage: HIDUsage.tab),
        ])
        let middle = UIStackView(arrangedSubviews: Self.symbols.map(characterButton)
            + (0..<Self.functionKeys).map { keyButton(title: "F\($0 + 1)", usage: HIDUsage.f1 + UInt16($0)) })
        let trailing = UIStackView(arrangedSubviews: [
            keyButton(symbol: "arrow.left", usage: HIDUsage.left),
            keyButton(symbol: "arrow.down", usage: HIDUsage.down),
            keyButton(symbol: "arrow.up", usage: HIDUsage.up),
            keyButton(symbol: "arrow.right", usage: HIDUsage.right),
            hideButton(),
        ])
        for stack in [leading, middle, trailing] { stack.spacing = Self.spacing }
        // The symbols and F-keys scroll; esc, ctrl, tab and the arrows stay put.
        let scroll = UIScrollView()
        scroll.showsHorizontalScrollIndicator = false
        scroll.addSubview(middle)
        middle.translatesAutoresizingMaskIntoConstraints = false
        let row = UIStackView(arrangedSubviews: [leading, scroll, trailing])
        row.spacing = Self.spacing
        row.translatesAutoresizingMaskIntoConstraints = false
        addSubview(row)
        // esc, ctrl, tab and the arrows keep their size; the scrolling strip takes what is left.
        for button in leading.arrangedSubviews + trailing.arrangedSubviews {
            button.widthAnchor.constraint(equalToConstant: Self.fixedKeyWidth).isActive = true
        }
        for stack in [leading, trailing] {
            stack.setContentHuggingPriority(.required, for: .horizontal)
            stack.setContentCompressionResistancePriority(.required, for: .horizontal)
        }
        scroll.setContentHuggingPriority(.defaultLow, for: .horizontal)
        scroll.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        NSLayoutConstraint.activate([
            row.leadingAnchor.constraint(equalTo: safeAreaLayoutGuide.leadingAnchor, constant: Self.spacing),
            row.trailingAnchor.constraint(equalTo: safeAreaLayoutGuide.trailingAnchor, constant: -Self.spacing),
            row.topAnchor.constraint(equalTo: topAnchor, constant: Self.spacing),
            row.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -Self.spacing),
            middle.leadingAnchor.constraint(equalTo: scroll.contentLayoutGuide.leadingAnchor),
            middle.trailingAnchor.constraint(equalTo: scroll.contentLayoutGuide.trailingAnchor),
            middle.topAnchor.constraint(equalTo: scroll.contentLayoutGuide.topAnchor),
            middle.bottomAnchor.constraint(equalTo: scroll.contentLayoutGuide.bottomAnchor),
            middle.heightAnchor.constraint(equalTo: scroll.frameLayoutGuide.heightAnchor),
        ])
    }

    required init?(coder: NSCoder) { fatalError("not used") }


    private func styled(title: String?, symbol: String?) -> UIButton {
        var configuration = UIButton.Configuration.gray()
        configuration.title = title
        configuration.image = symbol.flatMap { UIImage(systemName: $0) }
        configuration.cornerStyle = .medium
        configuration.contentInsets = NSDirectionalEdgeInsets(top: 4, leading: 8, bottom: 4, trailing: 8)
        let button = UIButton(configuration: configuration)
        button.configurationUpdateHandler = { button in
            button.configuration?.baseBackgroundColor = button.isSelected ? .tintColor : nil
            button.configuration?.baseForegroundColor = button.isSelected ? .white : .label
        }
        return button
    }

    private func keyButton(title: String? = nil, symbol: String? = nil, usage: UInt16) -> UIButton {
        let button = styled(title: title, symbol: symbol)
        button.addAction(UIAction { [weak self] _ in
            UIDevice.current.playInputClick()
            self?.onKey?(usage, true)
        }, for: .touchDown)
        for event: UIControl.Event in [.touchUpInside, .touchUpOutside, .touchCancel] {
            button.addAction(UIAction { [weak self] _ in self?.onKey?(usage, false) }, for: event)
        }
        return button
    }

    private func characterButton(_ character: Character) -> UIButton {
        let button = styled(title: String(character), symbol: nil)
        button.addAction(UIAction { [weak self] _ in
            UIDevice.current.playInputClick()
            self?.onCharacter?(character)
        }, for: .touchUpInside)
        return button
    }

    private func controlToggle() -> UIButton {
        let button = styled(title: nil, symbol: "control")
        button.addAction(UIAction { [weak self] _ in self?.controlLatched.toggle() }, for: .touchUpInside)
        controlButton = button
        return button
    }

    private func hideButton() -> UIButton {
        let button = styled(title: nil, symbol: "keyboard.chevron.compact.down")
        button.addAction(UIAction { [weak self] _ in self?.onHide?() }, for: .touchUpInside)
        return button
    }
}

extension KeyAccessory: UIInputViewAudioFeedback {
    var enableInputClicksWhenVisible: Bool { true }
}

final class ScreenUIView: UIView, UIKeyInput {
    weak var input: (any ScreenInput)?
    var surface: ScreenSurface? {
        didSet { attach() }
    }
    // The guest's image, in `contentRect`.
    private let imageLayer = CALayer()
    // The guest's pointer image, drawn where the pointer is: a touch has no host pointer to swap.
    private let pointerLayer = CALayer()
    // The image above the software keyboard rather than under it (MachineView's Keyboard Below
    // Screen). Follows the keyboard through `keyboardLayoutGuide`, accessory bar included.
    var keepsAboveKeyboard = false {
        didSet { if keepsAboveKeyboard != oldValue { setNeedsLayout() } }
    }
    // Pinned to the keyboard's top, so a keyboard that moves lays this view out again.
    private let keyboardTop = UIView()
    private var pointerPixel = CGPoint.zero
    private var scrollRemainder: CGFloat = 0
    // The view is first responder whenever it has been touched, so hardware keys reach the guest;
    // the software keyboard only shows when asked for (it covers half a phone's screen, and a
    // touch here is the pointer). An empty input view is how UIKit is told to show none.
    var showsSoftwareKeyboard = false {
        didSet {
            guard showsSoftwareKeyboard != oldValue else { return }
            if showsSoftwareKeyboard && !isFirstResponder { becomeFirstResponder() }
            reloadInputViews()
            setNeedsLayout()
        }
    }
    private let noKeyboard = UIView(frame: .zero)
    override var inputView: UIView? { showsSoftwareKeyboard ? nil : noKeyboard }
    private let accessory = KeyAccessory()
    override var inputAccessoryView: UIView? { showsSoftwareKeyboard ? accessory : nil }
    // The accessory's hide key: the SwiftUI side owns whether the keyboard shows.
    var onHideKeyboard: (() -> Void)?
    // A two-finger pan of this many points is one wheel notch.
    private static let pointsPerNotch: CGFloat = 12

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .black
        isMultipleTouchEnabled = true
        imageLayer.contentsGravity = .resizeAspect
        imageLayer.magnificationFilter = .nearest
        imageLayer.minificationFilter = .linear
        layer.addSublayer(imageLayer)
        keyboardTop.isUserInteractionEnabled = false
        keyboardTop.translatesAutoresizingMaskIntoConstraints = false
        addSubview(keyboardTop)
        keyboardLayoutGuide.followsUndockedKeyboard = true
        NSLayoutConstraint.activate([
            keyboardTop.leadingAnchor.constraint(equalTo: leadingAnchor),
            keyboardTop.widthAnchor.constraint(equalToConstant: 0),
            keyboardTop.heightAnchor.constraint(equalToConstant: 0),
            keyboardTop.bottomAnchor.constraint(equalTo: keyboardLayoutGuide.topAnchor),
        ])
        pointerLayer.isHidden = true
        pointerLayer.zPosition = 1
        layer.addSublayer(pointerLayer)
        addGestureRecognizer(UIHoverGestureRecognizer(target: self, action: #selector(hover(_:))))
        let pan = UIPanGestureRecognizer(target: self, action: #selector(twoFingerPan(_:)))
        pan.minimumNumberOfTouches = 2
        pan.allowedScrollTypesMask = .all
        addGestureRecognizer(pan)
        accessory.onKey = { [weak self] usage, pressed in self?.accessoryKey(usage, pressed: pressed) }
        accessory.onCharacter = { [weak self] character in self?.insertText(String(character)) }
        accessory.onHide = { [weak self] in self?.onHideKeyboard?() }
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    // A latched ctrl is held around the key and then let go.
    private func accessoryKey(_ usage: UInt16, pressed: Bool) {
        if pressed {
            if accessory.controlLatched { input?.key(usage: HIDUsage.leftControl, pressed: true) }
            input?.key(usage: usage, pressed: true)
        } else {
            input?.key(usage: usage, pressed: false)
            if accessory.controlLatched {
                input?.key(usage: HIDUsage.leftControl, pressed: false)
                accessory.controlLatched = false
            }
        }
    }

    private func attach() {
        surface?.onImage = { [weak self] in
            self?.imageLayer.contents = self?.surface?.image
            self?.placePointer()
        }
        surface?.onPointer = { [weak self] in self?.placePointer() }
        imageLayer.contents = surface?.image
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        imageLayer.frame = contentRect
        CATransaction.commit()
        placePointer()
    }

    // Where the image goes: all of the view, or the part above a shown keyboard.
    private var contentRect: CGRect {
        guard keepsAboveKeyboard, showsSoftwareKeyboard, keyboardTop.frame.maxY < bounds.maxY else { return bounds }
        return CGRect(x: bounds.minX, y: bounds.minY, width: bounds.width, height: max(0, keyboardTop.frame.maxY - bounds.minY))
    }

    private var imageSize: CGSize {
        CGSize(width: surface?.width ?? 0, height: surface?.height ?? 0)
    }

    private func placePointer() {
        guard case .image(let image, let hotX, let hotY) = surface?.pointer ?? .host, imageSize.width > 0 else {
            pointerLayer.isHidden = true
            return
        }
        let frame = AspectFit.frame(imageSize: imageSize, in: contentRect)
        let scale = frame.width / imageSize.width
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        pointerLayer.contents = image
        pointerLayer.frame = CGRect(x: frame.minX + (pointerPixel.x - CGFloat(hotX)) * scale,
                                    y: frame.minY + (pointerPixel.y - CGFloat(hotY)) * scale,
                                    width: CGFloat(image.width) * scale, height: CGFloat(image.height) * scale)
        pointerLayer.isHidden = false
        CATransaction.commit()
    }

    private func move(to point: CGPoint) {
        guard let pixel = AspectFit.pixel(at: point, imageSize: imageSize, in: contentRect) else { return }
        pointerPixel = CGPoint(x: pixel.x, y: pixel.y)
        input?.pointerMove(x: pixel.x, y: pixel.y)
        placePointer()
    }

    // Hardware keyboards (and the simulator's Mac keyboard): UIKey's code is the HID usage.
    override var canBecomeFirstResponder: Bool { true }

    override func pressesBegan(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        var unhandled = Set<UIPress>()
        for press in presses {
            if let key = press.key { input?.key(usage: UInt16(key.keyCode.rawValue), pressed: true) } else { unhandled.insert(press) }
        }
        if !unhandled.isEmpty { super.pressesBegan(unhandled, with: event) }
    }

    override func pressesEnded(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        var unhandled = Set<UIPress>()
        for press in presses {
            if let key = press.key { input?.key(usage: UInt16(key.keyCode.rawValue), pressed: false) } else { unhandled.insert(press) }
        }
        if !unhandled.isEmpty { super.pressesEnded(unhandled, with: event) }
    }

    override func pressesCancelled(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        pressesEnded(presses, with: event)
    }

    override func resignFirstResponder() -> Bool {
        input?.releaseAll()
        return super.resignFirstResponder()
    }

    // The software keyboard types characters, not keys: each becomes a press and release of the
    // key a US layout types it with, shift around it when needed.
    var hasText: Bool { true }

    func insertText(_ text: String) {
        let control = accessory.controlLatched
        if control { input?.key(usage: HIDUsage.leftControl, pressed: true) }
        for character in text {
            guard let (usage, shift) = HIDUsage.forCharacter(character) else { continue }
            if shift { input?.key(usage: HIDUsage.leftShift, pressed: true) }
            input?.key(usage: usage, pressed: true)
            input?.key(usage: usage, pressed: false)
            if shift { input?.key(usage: HIDUsage.leftShift, pressed: false) }
        }
        if control {
            input?.key(usage: HIDUsage.leftControl, pressed: false)
            accessory.controlLatched = false
        }
    }

    func deleteBackward() {
        input?.key(usage: HIDUsage.backspace, pressed: true)
        input?.key(usage: HIDUsage.backspace, pressed: false)
    }

    var autocorrectionType: UITextAutocorrectionType { get { .no } set {} }
    var autocapitalizationType: UITextAutocapitalizationType { get { .none } set {} }
    var smartQuotesType: UITextSmartQuotesType { get { .no } set {} }
    var smartDashesType: UITextSmartDashesType { get { .no } set {} }

    // A touch is the pointer and the left button: down where the finger lands, up where it lifts.
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        if !isFirstResponder { becomeFirstResponder() }
        guard touches.count == 1, let touch = touches.first else { return }
        move(to: touch.location(in: self))
        input?.pointerButton(0, pressed: true)
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        move(to: touch.location(in: self))
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        input?.pointerButton(0, pressed: false)
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        input?.pointerButton(0, pressed: false)
    }

    // A trackpad or mouse on an iPad moves the pointer without a button.
    @objc private func hover(_ recognizer: UIHoverGestureRecognizer) {
        move(to: recognizer.location(in: self))
    }

    @objc private func twoFingerPan(_ recognizer: UIPanGestureRecognizer) {
        // Dragging down scrolls up, as on the host.
        scrollRemainder += recognizer.translation(in: self).y
        recognizer.setTranslation(.zero, in: self)
        let notches = Int(scrollRemainder / Self.pointsPerNotch)
        scrollRemainder -= CGFloat(notches) * Self.pointsPerNotch
        if notches != 0 { input?.scroll(clicks: notches) }
    }
}

struct ScreenPane: UIViewRepresentable {
    let surface: ScreenSurface
    let input: any ScreenInput
    @Binding var showsSoftwareKeyboard: Bool
    var keepsAboveKeyboard = false

    func makeUIView(context: Context) -> ScreenUIView {
        let view = ScreenUIView(frame: .zero)
        view.surface = surface
        view.input = input
        return view
    }

    func updateUIView(_ view: ScreenUIView, context: Context) {
        view.input = input
        view.showsSoftwareKeyboard = showsSoftwareKeyboard
        view.keepsAboveKeyboard = keepsAboveKeyboard
        view.onHideKeyboard = { showsSoftwareKeyboard = false }
    }
}
#endif
