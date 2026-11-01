// A service worker that makes the page cross-origin isolated where the server cannot (GitHub
// Pages sends no custom headers). SharedArrayBuffer, which carries keystrokes, resizes and the stop
// flag into the emulator's worker, exists only in an isolated page: COOP `same-origin` and COEP
// `require-corp` on the document, and COEP on the worker script. This adds them to every response
// it passes on, and CORP `cross-origin` so the xterm scripts from the CDN may load under COEP.
// isolate.js registers it only when the page is not isolated already; hack/web/serve.py sends the
// headers itself.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const request = event.request;
  // Chrome rejects these outside same-origin mode instead of answering them.
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
  // Disk reads are range requests from the worker; their 206 goes to it untouched.
  if (request.headers.has("range")) return;
  event.respondWith(fetch(request).then((response) => {
    // Opaque responses (no-cors, redirects) cannot be read, so they cannot be rewrapped.
    if (response.status === 0) return response;
    const headers = new Headers(response.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    headers.set("Cross-Origin-Resource-Policy", "cross-origin");
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  }));
});
