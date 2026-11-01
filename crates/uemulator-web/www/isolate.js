// Runs before anything else on the page. A page the server did not make cross-origin isolated
// (GitHub Pages) registers isolation_worker.js and reloads once it takes control, so the page comes
// back through it with the headers. A page that is controlled and still not isolated stops there:
// reloading again would loop forever in a browser where the worker cannot help.
if (!self.crossOriginIsolated && self.isSecureContext && "serviceWorker" in navigator) {
  navigator.serviceWorker.register("isolation_worker.js").then(() => {
    if (navigator.serviceWorker.controller) return;
    navigator.serviceWorker.addEventListener("controllerchange", () => location.reload(), { once: true });
  }).catch((error) => console.warn("uemulator: cannot register isolation_worker.js:", error));
}
