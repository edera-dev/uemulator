import SwiftUI

// Where a half-open foldable bends, measured from the top of the machine's view. MachineView puts
// the screen or terminal above the fold and FoldedControls below it.
struct FoldPosture: Equatable {
    var foldY: CGFloat

    // Kept clear on each side of the fold: the crease distorts the picture and is awkward to touch.
    static let clearance: CGFloat = 12
}

#if os(iOS)
import UIKit

// Reports the fold while the device is partially open in portrait, nil otherwise (fully open,
// closed, landscape, or a device without a hinge). Placed behind the machine's whole view, so
// `foldY` is in its coordinates.
struct HingeReader: UIViewRepresentable {
    @Binding var posture: FoldPosture?

    func makeUIView(context: Context) -> HingeReaderView {
        HingeReaderView(frame: .zero)
    }

    func updateUIView(_ view: HingeReaderView, context: Context) {
        view.onChange = { posture = $0 }
    }
}

final class HingeReaderView: UIView {
    var onChange: ((FoldPosture?) -> Void)?
    private var partiallyOpen = false
    private var reported: FoldPosture?
    // `-fold-posture partial` on the command line: a simulator has no hinge to bend, so this is how
    // the folded layout is tried there.
    private static let simulated = UserDefaults.standard.string(forKey: "fold-posture") == "partial"

    override init(frame: CGRect) {
        super.init(frame: frame)
        isUserInteractionEnabled = false
        if #available(iOS 27.1, *) {
            addInteraction(UIHingeInteraction { [weak self] _, update in
                self?.partiallyOpen = update.hinge?.status == .partiallyOpen
                self?.report()
            })
        }
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    override func layoutSubviews() {
        super.layoutSubviews()
        report()
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        report()
    }

    // UIHinge gives no position. The fold is taken to cross the middle of the screen's long side:
    // in portrait that is a horizontal line halfway down, which is the only posture the folded
    // layout is for (a fold down the middle of a landscape screen keeps the usual layout).
    private var posture: FoldPosture? {
        guard partiallyOpen || Self.simulated, let screen = window?.windowScene?.screen else { return nil }
        let screenBounds = screen.bounds
        guard screenBounds.height > screenBounds.width else { return nil }
        let foldY = convert(CGPoint(x: 0, y: screenBounds.midY), from: screen.coordinateSpace).y
        guard foldY > 0, foldY < bounds.height else { return nil }
        return FoldPosture(foldY: foldY.rounded())
    }

    private func report() {
        let current = posture
        guard current != reported else { return }
        reported = current
        // Layout runs inside SwiftUI's own update, where changing its state is not allowed.
        DispatchQueue.main.async { [weak self] in self?.onChange?(current) }
    }
}

// The half below the fold: everything that is input or control, so none of it covers the guest.
// The controls sit just under the fold, or on top of the keyboard while one is up, so keys and
// controls are one block; a keyboard taller than the half hides them (clipped, never drawn over
// the guest) until its hide key. Full screen adds what the hidden navigation bar had (state,
// settings, start and stop).
struct FoldedControls: View {
    @ObservedObject var model: MachineModel
    @Binding var fullscreen: Bool
    @Binding var softwareKeyboard: Bool
    // What Resize to Fit asks the guest for: the part above the fold.
    let screenSize: CGSize
    let onSettings: () -> Void
    @Environment(\.displayScale) private var displayScale
    // SwiftTerm's keyboard is its own (the terminal view is first responder); this follows it.
    @State private var terminalKeyboard = false
    // Window coordinates (SwiftUI's global space); the keyboard's frame is in screen coordinates,
    // the same thing for the full-screen window a phone gives an app.
    @State private var panelBottom: CGFloat = 0
    @State private var keyboardFrame = CGRect.null

    private static let spacing: CGFloat = 12
    // Shorter than any software keyboard, taller than a hardware keyboard's shortcut bar: a
    // keyboard frame at least this tall is the software keyboard showing.
    private static let softwareKeyboardMinHeight: CGFloat = 150

    private var keyboardShown: Bool {
        model.view == .screen ? softwareKeyboard : terminalKeyboard
    }

    // How much of the panel's bottom the keyboard (or a hardware keyboard's shortcut bar) covers.
    private var keyboardOverlap: CGFloat {
        guard !keyboardFrame.isNull else { return 0 }
        return max(0, panelBottom - keyboardFrame.minY.rounded())
    }

    // The keys first, just below the fold and outside the panel the keyboard clips, so they show
    // whatever the keyboard's height. They are the screen's or the terminal's own (the one that
    // receives typing), taken off the keyboard while folded.
    var body: some View {
        VStack(spacing: 0) {
            FoldedKeyBar(bar: model.view == .screen ? model.screenFoldedKeyBar : model.terminal.foldedKeyBar)
                .frame(height: KeyAccessory.barHeight)
            controls
        }
        // The guest stops `clearance` above the fold (MachineView); this starts as far below it.
        .padding(.top, 2 * FoldPosture.clearance)
        .background(Color.black)
    }

    private var controls: some View {
        VStack(spacing: Self.spacing) {
            HStack(spacing: Self.spacing) {
                if model.definition.display.hasScreen {
                    Picker("View", selection: $model.view) {
                        Label("Terminal", systemImage: "terminal").tag(MachineModel.View.terminal)
                        Label("Screen", systemImage: "display").tag(MachineModel.View.screen)
                    }
                    .pickerStyle(.segmented)
                    .fixedSize()
                }
                Spacer(minLength: 0)
                Button(keyboardShown ? "Hide Keyboard" : "Keyboard",
                       systemImage: keyboardShown ? "keyboard.chevron.compact.down" : "keyboard") {
                    setKeyboard(!keyboardShown)
                }
                if model.view == .screen {
                    Button("Resize to Fit", systemImage: "aspectratio") {
                        model.fitScreen(to: screenSize, scale: displayScale)
                    }
                    .disabled(!model.isRunning)
                }
                Button(fullscreen ? "Exit Full Screen" : "Full Screen",
                       systemImage: fullscreen ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right") {
                    fullscreen.toggle()
                }
            }
            .labelStyle(.iconOnly)
            if fullscreen {
                HStack(spacing: Self.spacing) {
                    StateChip(state: model.state, startedAt: model.startedAt)
                        .labelStyle(.titleAndIcon)
                        .lineLimit(1)
                    Spacer(minLength: 0)
                    Button("Settings", systemImage: "slider.horizontal.3", action: onSettings)
                        .labelStyle(.iconOnly)
                    if model.isRunning {
                        Button("Stop", systemImage: "stop.fill") { model.stop() }
                    } else {
                        Button(model.startTitle, systemImage: "play.fill") { model.start() }
                    }
                }
            }
            if !keyboardShown {
                // The rest of the half, empty while the keyboard is down: one tap brings it up.
                Button { setKeyboard(true) } label: {
                    Label("Tap to Type", systemImage: "keyboard")
                        .font(.title3)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }
        }
        .buttonStyle(.bordered)
        .controlSize(.large)
        .padding(.top, Self.spacing)
        .padding(.horizontal)
        .padding(.bottom, keyboardOverlap)
        // minHeight 0: the half's height whatever the content asks for. Without it a tall keyboard's
        // padding grows the panel, which moves its bottom, which changes the padding: a layout loop
        // that hangs the main thread.
        .frame(maxWidth: .infinity, minHeight: 0, maxHeight: .infinity, alignment: keyboardOverlap > 0 ? .bottom : .top)
        .clipped()
        .background(.bar)
        .onGeometryChange(for: CGFloat.self) { $0.frame(in: .global).maxY.rounded() } action: { panelBottom = $0 }
        .onReceive(NotificationCenter.default.publisher(for: UIResponder.keyboardWillChangeFrameNotification)) { note in
            keyboardFrame = note.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? CGRect ?? .null
        }
        .onReceive(NotificationCenter.default.publisher(for: UIResponder.keyboardWillShowNotification)) { note in
            let frame = note.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? CGRect ?? .zero
            terminalKeyboard = frame.height >= Self.softwareKeyboardMinHeight && model.terminal.view.isFirstResponder
        }
        .onReceive(NotificationCenter.default.publisher(for: UIResponder.keyboardWillHideNotification)) { _ in
            terminalKeyboard = false
            keyboardFrame = .null
        }
    }

    private func setKeyboard(_ shown: Bool) {
        switch model.view {
        case .screen:
            softwareKeyboard = shown
        case .terminal:
            if shown {
                _ = model.terminal.view.becomeFirstResponder()
            } else {
                _ = model.terminal.view.resignFirstResponder()
            }
        }
    }
}

// Hosts one of the freestanding key bars in a container SwiftUI owns, so swapping between the
// screen's and the terminal's moves only the bar, never a view SwiftUI is laying out.
struct FoldedKeyBar: UIViewRepresentable {
    let bar: KeyAccessory

    func makeUIView(context: Context) -> UIView {
        let container = UIView()
        container.backgroundColor = .black
        return container
    }

    func updateUIView(_ container: UIView, context: Context) {
        guard bar.superview !== container else { return }
        container.subviews.forEach { $0.removeFromSuperview() }
        bar.frame = container.bounds
        bar.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        container.addSubview(bar)
    }
}
#endif
