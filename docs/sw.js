// Offline cache + cross-origin isolation. GitHub Pages can't set COOP/COEP headers, which WASM
// threads need (SharedArrayBuffer), so every same-origin response goes out with them added here.
// Bump VERSION whenever models or vendor files change: those are served cache-first.
const VERSION = "v1";
const CACHE = `cjk-reader-${VERSION}`;
const SHELL = [
  "./", "index.html", "style.css", "app.js", "ocr-worker.js", "ocr-core.js", "reading-worker.js", "reading.js",
  "manifest.webmanifest", "icons/icon-192.png", "icons/apple-touch-icon.png",
  "vendor/pinyin-pro.js", "vendor/wanakana.min.js",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== CACHE) await caches.delete(k);
    await self.clients.claim();
  })());
});

function isolate(res) {
  if (res.type === "opaque" || res.status === 0) return res;
  const headers = new Headers(res.headers);
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
}

// Models, ORT and dictionaries are large and versioned: cache-first.
async function cacheFirst(req) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok) await cache.put(req, res.clone());
  return res;
}

// App code: network-first so deploys show up, cache when offline.
async function networkFirst(req) {
  const cache = await caches.open(CACHE);
  try {
    const res = await fetch(req);
    if (res.ok) await cache.put(req, res.clone());
    return res;
  } catch (err) {
    const hit = await cache.match(req, { ignoreSearch: req.mode === "navigate" });
    if (hit) return hit;
    throw err;
  }
}

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  const heavy = /\/(models|vendor)\//.test(url.pathname);
  e.respondWith((heavy ? cacheFirst : networkFirst)(e.request).then(isolate));
});
