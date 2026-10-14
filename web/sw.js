// Tiny offline cache so the page opens even with no signal in the lab.
// Network first for everything: the phone always gets the newest page when it has signal,
// and falls back to the cached copy when it does not. VERSION must match web/version.js.
const VERSION = "v2";
const CACHE = "sawmeter-" + VERSION;
const FILES = ["./", "./index.html", "./style.css", "./app.js", "./audio.js", "./features.js", "./detector.js",
  "./version.js", "./icon.svg", "./manifest.webmanifest"];

self.addEventListener("install", (e) => e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting())));
self.addEventListener("activate", (e) => e.waitUntil(
  caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(fetch(e.request, { cache: "no-cache" })
    .then((r) => { if (r.ok) caches.open(CACHE).then((c) => c.put(e.request, r.clone())); return r; })
    .catch(() => caches.match(e.request)));
});
