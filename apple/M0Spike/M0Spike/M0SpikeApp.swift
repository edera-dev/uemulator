import SwiftUI
import WebKit

@main
struct M0SpikeApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @StateObject private var engine = Engine()

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("uemulator M0").font(.headline)
                Spacer()
                Button("Run M0") { Task { await engine.runAll() } }.disabled(engine.busy)
            }
            ScrollView {
                Text(engine.log.joined(separator: "\n"))
                    .font(.system(.caption2, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(maxHeight: 220)
            ScrollViewReader { proxy in
                ScrollView {
                    Text(engine.console)
                        .font(.system(size: 9, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .id("end")
                }
                .onChange(of: engine.console) { proxy.scrollTo("end", anchor: .bottom) }
            }
            .background(Color.black.opacity(0.05))
            // The engine's web view: in the hierarchy (WebKit may stop a page whose view is in no
            // window) but hidden and 1x1. It is never what the user looks at.
            HiddenWebView(webView: engine.webView).frame(width: 1, height: 1).opacity(0)
        }
        .padding()
        .task {
            if ProcessInfo.processInfo.arguments.contains("-autorun") { await engine.runAll() }
        }
    }
}

struct HiddenWebView: UIViewRepresentable {
    let webView: WKWebView
    func makeUIView(context: Context) -> WKWebView { webView }
    func updateUIView(_ uiView: WKWebView, context: Context) {}
}
