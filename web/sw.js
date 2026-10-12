// Tiny offline cache so the page opens even with no signal in the lab.
// The page files are cached; model.json is always fetched fresh when online.
const CACHE = "sawmeter-v1";
const FILES = ["./", "./index.html", "./style.css", "./app.js", "./audio.js", "./features.js", "./detector.js", "./icon.svg", "./manifest.webmanifest"];

self.addEventListener("install", (e) => e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting())));
self.addEventListener("activate", (e) => e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))));
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.pathname.endsWith("model.json")) {
    e.respondWith(fetch(e.request).then((r) => { caches.open(CACHE).then((c) => c.put(e.request, r.clone())); return r; })
      .catch(() => caches.match(e.request)));
  } else {
    e.respondWith(caches.match(e.request).then((r) => r || fetch(e.request)));
  }
});
