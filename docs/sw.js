// Offline cache + cross-origin isolation. GitHub Pages can't set COOP/COEP headers, which WASM
// threads need (SharedArrayBuffer), so every same-origin response goes out with them added here.
//
// App code is revalidated with the server on every load, so deploys show up without any version
// bump. Models/vendor files are large and served cache-first: bump ASSETS when they change.
const SHELL = "cjk-reader-shell-v2";
const ASSETS = "cjk-reader-v2"; // v2: PP-OCRv6 models
const SHELL_FILES = [
  "./", "index.html", "style.css", "app.js", "ocr-worker.js", "ocr-core.js", "reading-worker.js", "reading.js",
  "manifest.webmanifest", "icons/icon-192.png", "icons/apple-touch-icon.png",
  "vendor/pinyin-pro.js", "vendor/wanakana.min.js",
];

self.addEventListener("install", (e) => {
  // cache: "reload" skips the HTTP cache, which GitHub Pages lets hold files for 10 minutes.
  e.waitUntil(caches.open(SHELL)
    .then((c) => c.addAll(SHELL_FILES.map((f) => new Request(f, { cache: "reload" }))))
    .then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== SHELL && k !== ASSETS) await caches.delete(k);
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

// Models, ORT and dictionaries: cache-first. A download is stored in the background while it streams
// to the page; waiting for the cache first would hold back every byte (and the progress bar) until
// the whole file had arrived.
async function cacheFirst(req, event) {
  const hit = await caches.match(req); // any cache: the small vendor scripts are precached in SHELL
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok) event.waitUntil(caches.open(ASSETS).then((c) => c.put(req, res.clone())));
  return res;
}

// App code: always ask the server (a cheap 304 when unchanged), fall back to the cache offline.
async function networkFirst(req) {
  const cache = await caches.open(SHELL);
  try {
    // A navigation Request can't be re-initialised with options, so fetch its URL instead.
    const res = await (req.mode === "navigate" ? fetch(req.url, { cache: "no-cache" }) : fetch(req, { cache: "no-cache" }));
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
  e.respondWith((heavy ? cacheFirst(e.request, e) : networkFirst(e.request)).then(isolate));
});
