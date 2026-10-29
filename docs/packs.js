// Language packs: each language's reading files and its translation files (docs/packs.json), downloaded
// when the settings panel turns them on, checked against their sha256 and kept in the service worker's
// cache for models. Files already cached (an earlier download, or the reader fetching them) are skipped.

export const CACHE = "cjk-reader-v2"; // sw.js ASSETS: keep the two in step

let manifest = null;
function load() {
  manifest ??= fetch("packs.json")
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`packs.json: HTTP ${r.status}`))))
    .catch((e) => { manifest = null; throw e; });
  return manifest;
}
export const packs = () => load().then((m) => m.packs);
// Paths the app needs whatever is installed (the OCR runtime), never removed with a pack.
export const core = () => load().then((m) => new Set(m.core));

export const bytes = (files) => files.reduce((a, f) => a + f.size, 0);
const url = (f) => new URL(f.path, location.href).href;
const hex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("");

// The files of a pack not in any cache yet.
export async function missing(files) {
  if (!self.caches) return files;
  const hits = await Promise.all(files.map((f) => caches.match(url(f))));
  return files.filter((f, i) => !hits[i]);
}

// Download a pack's missing files one at a time (one large buffer in memory at once), calling
// onProgress(loaded, total) on the way. A file is cached only once its sha256 matches, so a failed or
// corrupted download leaves nothing behind but the files before it, and the next try fetches the rest.
export async function install(files, onProgress = () => {}) {
  const todo = await missing(files);
  const total = bytes(todo);
  if (!total) return;
  const { usage = 0, quota = Infinity } = (await navigator.storage?.estimate?.()) ?? {};
  if (quota - usage < 1.2 * total && !confirm(`This needs ${mb(total)} and the device may not have room. Download anyway?`))
    throw new Error("Not enough storage");
  const cache = await caches.open(CACHE);
  let done = 0, last = 0;
  for (const f of todo) {
    // no-store: sw.js passes it straight to the network instead of caching it unchecked.
    const res = await fetch(url(f), { cache: "no-store" });
    if (!res.ok) throw new Error(`${f.path}: HTTP ${res.status}`);
    const buf = new Uint8Array(f.size);
    let o = 0;
    const reader = res.body.getReader();
    for (let r; !(r = await reader.read()).done;) {
      if (o + r.value.length > buf.length) throw new Error(`${f.path}: larger than expected`);
      buf.set(r.value, o);
      o += r.value.length;
      if (performance.now() - last > 100) {
        last = performance.now();
        onProgress(done + o, total);
      }
    }
    if (o !== f.size || hex(await crypto.subtle.digest("SHA-256", buf)) !== f.sha256) throw new Error(`${f.path}: download corrupted`);
    await cache.put(url(f), new Response(buf, { headers: {
      "Content-Type": res.headers.get("Content-Type") || "application/octet-stream",
      "Content-Length": String(f.size), // the workers' progress bars read it
    } }));
    done += f.size;
    onProgress(done, total);
  }
  // Ask the browser not to clear the downloads under storage pressure (granted silently or not at all).
  navigator.storage?.persist?.().catch(() => {});
}

// Delete a pack's files, except those another installed pack still needs (keep: their paths).
export async function remove(files, keep = new Set()) {
  if (!self.caches) return;
  for (const key of await caches.keys()) {
    const cache = await caches.open(key);
    for (const f of files) if (!keep.has(f.path)) await cache.delete(url(f));
  }
}

// Delete cached files under a folder ("models/translate/") that no installed pack lists any more: an
// update that changed a language's translator leaves its old model behind otherwise.
export async function prune(folder, keep) {
  if (!self.caches) return;
  const base = new URL(folder, location.href).href;
  for (const key of await caches.keys()) {
    const cache = await caches.open(key);
    for (const req of await cache.keys())
      if (req.url.startsWith(base) && !keep.has(req.url.slice(new URL(".", location.href).href.length))) await cache.delete(req);
  }
}

export async function usage() {
  return (await navigator.storage?.estimate?.())?.usage ?? null;
}

export const mb = (n) => `${(n / 1e6).toFixed(n < 1e7 ? 1 : 0)} MB`;
