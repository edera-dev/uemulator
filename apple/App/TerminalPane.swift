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

    override init() {
        super.init()
        view.terminalDelegate = self
        #if os(macOS)
        view.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
        #else
        view.font = UIFont.monospacedSystemFont(ofSize: 11, weight: .regular)
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
        MainActor.assumeIsolated { onInput?(bytes) }
    }

    nonisolated func sizeChanged(source: TerminalView, newCols: Int, newRows: Int) {
        MainActor.assumeIsolated {
            size = (newCols, newRows)
            onResize?(newCols, newRows)
        }
    }

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
