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

final class ScreenUIView: UIView, UIKeyInput {
    weak var input: (any ScreenInput)?
    var surface: ScreenSurface? {
        didSet { attach() }
    }
    // The guest's pointer image, drawn where the pointer is: a touch has no host pointer to swap.
    private let pointerLayer = CALayer()
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
        }
    }
    private let noKeyboard = UIView(frame: .zero)
    override var inputView: UIView? { showsSoftwareKeyboard ? nil : noKeyboard }
    // A two-finger pan of this many points is one wheel notch.
    private static let pointsPerNotch: CGFloat = 12

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .black
        isMultipleTouchEnabled = true
        layer.contentsGravity = .resizeAspect
        layer.magnificationFilter = .nearest
        layer.minificationFilter = .linear
        pointerLayer.isHidden = true
        pointerLayer.zPosition = 1
        layer.addSublayer(pointerLayer)
        addGestureRecognizer(UIHoverGestureRecognizer(target: self, action: #selector(hover(_:))))
        let pan = UIPanGestureRecognizer(target: self, action: #selector(twoFingerPan(_:)))
        pan.minimumNumberOfTouches = 2
        pan.allowedScrollTypesMask = .all
        addGestureRecognizer(pan)
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    private func attach() {
        surface?.onImage = { [weak self] in
            self?.layer.contents = self?.surface?.image
            self?.placePointer()
        }
        surface?.onPointer = { [weak self] in self?.placePointer() }
        layer.contents = surface?.image
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        placePointer()
    }

    private var imageSize: CGSize {
        CGSize(width: surface?.width ?? 0, height: surface?.height ?? 0)
    }

    private func placePointer() {
        guard case .image(let image, let hotX, let hotY) = surface?.pointer ?? .host, imageSize.width > 0 else {
            pointerLayer.isHidden = true
            return
        }
        let frame = AspectFit.frame(imageSize: imageSize, in: bounds)
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
        guard let pixel = AspectFit.pixel(at: point, imageSize: imageSize, in: bounds) else { return }
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
        for character in text {
            guard let (usage, shift) = HIDUsage.forCharacter(character) else { continue }
            if shift { input?.key(usage: HIDUsage.leftShift, pressed: true) }
            input?.key(usage: usage, pressed: true)
            input?.key(usage: usage, pressed: false)
            if shift { input?.key(usage: HIDUsage.leftShift, pressed: false) }
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
    var showsSoftwareKeyboard = false

    func makeUIView(context: Context) -> ScreenUIView {
        let view = ScreenUIView(frame: .zero)
        view.surface = surface
        view.input = input
        return view
    }

    func updateUIView(_ view: ScreenUIView, context: Context) {
        view.input = input
        view.showsSoftwareKeyboard = showsSoftwareKeyboard
    }
}
#endif
