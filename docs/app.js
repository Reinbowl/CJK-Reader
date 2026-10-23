// Camera -> OCR worker -> reading worker -> overlay of pinyin / romaji / romanization above the text.
const $ = (s) => document.querySelector(s);
const view = $("#view"), video = $("#cam"), still = $("#still"), overlay = $("#overlay");
const statusEl = $("#status"), sheet = $("#sheet"), linesEl = $("#lines");

// URL switches for performance testing: ?ep=webgpu &precision=fp32 &threads=1 &det=480 &spin=1 &debug=1.
// A home-screen app can't be given URL switches, so ep/threads/spin/debug can also be set in the dev
// panel (tap the status line three times), which keeps them in localStorage. A URL switch wins.
const params = new URLSearchParams(location.search);
const dev = (() => { try { return JSON.parse(localStorage.getItem("dev")) || {}; } catch { return {}; } })();
const setting = (k) => params.get(k) ?? dev[k];
const EP = setting("ep") === "webgpu" ? "webgpu" : "wasm";
const PRECISION = params.get("precision") === "fp32" ? "fp32" : "int8";
const THREADS = Number(setting("threads")) || (self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1);
const SPIN = String(setting("spin")) === "1"; // let ORT's threads busy-wait between ops (its own default)
let DEBUG = params.has("debug") || !!dev.debug;
const LIVE_DET = Number(params.get("det")) || 640; // detector input, longest side
const STILL_DET = 960;
const LIVE_MAX = 1280, STILL_MAX = 1920; // captured image, longest side (recogniser crops come from this)
const MODEL = { zh: "zhja", ja: "zhja", ko: "ko" };
// Labels: LABEL_SCALE × the line's thickness, at most MAX_LABEL px, shrunk to fit. A row fades in
// between FADE_FROM and LEGIBLE px on screen (pinching a frozen frame enlarges it); until it's fully
// shown, its line's box is dashed: zoom in. Live mode doesn't recognise lines thinner than MIN_LINE
// px on screen (dev panel: 10/14/20).
const LABEL_SCALE = 0.45, MAX_LABEL = 28, FADE_FROM = 4, LEGIBLE = 6;
const OVERLAY_MIN = 14; // px: rows this big go over their own text rather than shrink below 3/4
const MIN_LINE = Number(setting("minline")) || 14;
const MIN_GAP = 250; // live mode: ms between reads, at least, and at least as long as the last read took

// What the app is doing, for the e2e tests (test/e2e.test.mjs) and ?debug.
const stats = (self.cjkStats = { mode: "idle", backend: "", reads: 0, lines: 0, skipped: 0, recognised: 0, busy: 0, ms: null, gate: null, fast: null, labels: null, reading: 0, cancelled: 0, zoom: null, download: null });

let lang = (() => {
  try { const saved = localStorage.getItem("lang"); if (MODEL[saved]) return saved; } catch {}
  const nav = (navigator.language || "").slice(0, 2);
  return MODEL[nav] ? nav : "zh";
})();
let mode = "idle"; // idle | live | still
let stream = null, loopToken = 0, lastStill = null;
// Zoom. Live: `zoom` = the camera's own zoom (hwZoom, if the browser exposes it) × digital `dzoom`.
// Still: `sv` is the pinch/pan transform of the frozen frame (screen = sv.s × base + sv.t).
let zoom = 1, dzoom = 1, hwZoom = null, sv = { s: 1, tx: 0, ty: 0 };

// ---------- service worker (offline + cross-origin isolation for WASM threads) ----------
let reloading = false;
if ("serviceWorker" in navigator) {
  // A new sw.js took over a page the old one served: reload once so all code comes from the new one.
  if (navigator.serviceWorker.controller) {
    navigator.serviceWorker.addEventListener("controllerchange", () => { reloading = true; location.reload(); }, { once: true });
  }
  navigator.serviceWorker.register("sw.js").then(() => {
    // GitHub Pages can't send COOP/COEP headers; sw.js adds them, but only to pages it controls.
    let tried = false;
    try { tried = sessionStorage.getItem("coi-reload") === "1"; } catch {}
    if (self.crossOriginIsolated || tried) return;
    const reload = () => {
      try { sessionStorage.setItem("coi-reload", "1"); } catch {}
      reloading = true;
      location.reload();
    };
    if (navigator.serviceWorker.controller) reload();
    else navigator.serviceWorker.addEventListener("controllerchange", reload, { once: true });
  });
}

// ---------- workers ----------
function rpc(worker, onEvent) {
  let seq = 0;
  const pending = new Map();
  worker.onmessage = ({ data }) => {
    const p = pending.get(data.id);
    if (!p) return onEvent(data);
    pending.delete(data.id);
    data.error ? p.reject(Object.assign(new Error(data.error), { where: data.where, cancelled: data.cancelled })) : p.resolve(data);
  };
  worker.onerror = (e) => setStatus(`Worker error: ${e.message}`, true);
  return (msg, transfer = []) => new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...msg, id }, transfer);
  });
}

// Downloads (runtime, models, the Japanese dictionary) as one progress bar on the start screen, and
// in the status line if the camera is already running. A new batch starts once the last one is done.
const progress = {};
const finished = (p) => p.total && p.loaded >= p.total;
function onProgress({ type, file, loaded, total }) {
  if (type !== "progress") return;
  if (!progress[file] && Object.values(progress).every(finished)) for (const f in progress) delete progress[f];
  progress[file] = { loaded, total: Math.max(total, 0) };
  const files = Object.values(progress), done = files.every(finished);
  const sum = (f) => files.reduce((a, p) => a + f(p), 0);
  const all = sum((p) => p.total || p.loaded), got = sum((p) => Math.min(p.loaded, p.total || p.loaded));
  const pct = all ? (100 * got) / all : 0;
  stats.download = { pct, done, ready: false };
  $("#progress").hidden = false;
  $("#progress i").style.width = `${pct.toFixed(1)}%`;
  $("#progress p").textContent = done ? "Preparing…" : `Downloading (first time only) · ${(got / 1e6).toFixed(1)} / ${(all / 1e6).toFixed(1)} MB`;
  if (mode === "live") setStatus(done ? "Preparing…" : `Downloading… ${Math.floor(pct)} %`);
}

// The current language's models are loaded: only now is the reader ready.
function showReady() {
  stats.download = { ...stats.download, ready: true };
  if ($("#progress").hidden) return; // nothing was downloaded: everything came from the cache quietly
  $("#progress i").style.width = "100%";
  $("#progress p").textContent = "Ready";
}

// The OCR worker also says when it had to move from WebGPU to WASM mid-session.
const ocr = rpc(new Worker("ocr-worker.js", { type: "module" }), (d) => (d.type === "backend" ? onBackend(d) : onProgress(d)));
const reading = rpc(new Worker("reading-worker.js"), onProgress);
// The worker starts on the runtime, the detector and this language's recogniser at once. If that
// first download fails (a dropped connection), the next read tries again.
let inited;
function init() {
  inited = ocr({ type: "init", ep: EP, precision: PRECISION, threads: THREADS, spin: SPIN, model: MODEL[lang] });
  inited.then(onBackend, () => {});
}
init();
let backend = EP;
function onBackend(r) {
  backend = r.fallback ? `${r.backend} (no webgpu)` : r.backend;
  stats.backend = { ...r };
  if (r.fallback) console.warn(r.fallback);
  debugReadout();
}
const loaded = {};
function ensureModels(l) {
  loaded[l] ??= Promise.all([inited, ocr({ type: "load", model: MODEL[l] }), reading({ type: "load", lang: l })])
    .catch((e) => {
      delete loaded[l];
      const failed = inited;
      inited.catch(() => { if (inited === failed) init(); });
      throw e;
    });
  return loaded[l];
}

// ---------- capture ----------
const grabCanvas = document.createElement("canvas");
const grabCtx = grabCanvas.getContext("2d", { willReadFrequently: true });

// The part of a srcW×srcH source that is on screen. fit "cover" matches the video's CSS (centre
// crop), magnified z times by digital zoom; "contain" shows a whole photo.
function visibleRect(srcW, srcH, fit, z = 1) {
  if (fit !== "cover") return { sx: 0, sy: 0, sw: srcW, sh: srcH };
  const s = Math.max(view.clientWidth / srcW, view.clientHeight / srcH) * z;
  const sw = view.clientWidth / s, sh = view.clientHeight / s;
  return { sx: (srcW - sw) / 2, sy: (srcH - sh) / 2, sw, sh };
}

// Draw the part of `src` that is on screen into grabCanvas (at most maxSide px) and return its
// pixels, the image->screen mapping, and the source rectangle and scale (to redraw it later).
function grab(src, srcW, srcH, maxSide, fit, z = 1) {
  const vw = view.clientWidth, vh = view.clientHeight;
  const rect = visibleRect(srcW, srcH, fit, z), { sx, sy, sw, sh } = rect;
  const k = Math.min(1, maxSide / Math.max(sw, sh));
  const w = Math.round(sw * k), h = Math.round(sh * k);
  grabCanvas.width = w; grabCanvas.height = h;
  grabCtx.drawImage(src, sx, sy, sw, sh, 0, 0, w, h);
  const s = Math.min(vw / w, vh / h);
  // A whole photo sits just under the header so the results sheet covers less of it.
  const oy = fit === "contain" ? Math.min((vh - h * s) / 2, $("header").offsetHeight) : (vh - h * s) / 2;
  return { image: grabCtx.getImageData(0, 0, w, h), map: { s, ox: (vw - w * s) / 2, oy }, rect, k };
}

// minLineH: image px; thinner lines are left unrecognised and come back in `skipped`.
async function read(image, detLimit, minLineH = 0) {
  await ensureModels(lang);
  const l = lang;
  stats.reading++;
  const res = await ocr({ type: "ocr", model: MODEL[l], detLimit, minLineH, buffer: image.data.buffer, width: image.width, height: image.height }, [image.data.buffer])
    .catch((e) => { if (e.cancelled) stats.cancelled++; throw e; })
    .finally(() => stats.reading--);
  const notes = res.lines.length ? (await reading({ type: "annotate", lang: l, texts: res.lines.map((x) => x.text) })).lines : [];
  Object.assign(stats, { reads: stats.reads + 1, lines: res.lines.length, skipped: res.skipped.length, recognised: stats.recognised + res.recognised, ms: res.ms });
  return { ...res, notes, lang: l };
}

// ---------- motion gate (live mode) ----------
// Every GATE_MS the on-screen part of the video is shrunk to a grey thumbnail THUMB_W px wide,
// normalised (mean 0, sd 1) so exposure changes don't count, and compared with the previous one
// (motion) and with the one taken when the overlay's frame was read (drift).
// - A fast move (or a different scene) fades the overlay out.
// - Otherwise the overlay follows the drift, so hand tremor doesn't hide the labels.
// - A read starts once the view has been steady for SETTLE samples and has drifted more than
//   CHANGE_PX or shows something else.
// "Same scene" is judged against the noise seen while steady: big sharp text leaves more residue
// after a sub-pixel shift than a soft scene, and the tolerance grows with it.
// Zoomed in, the same hand shake moves the picture further, so the thumbnail gets coarser in step
// (THUMB_W / zoom px wide), and the thresholds are in px of the 1× thumbnail: they stand for the
// same movement of the phone at any zoom. Past 2.5× the thumbnail stops at MIN_THUMB px and its
// shifts are scaled by `u` instead. The shift search reaches SEARCH px of that movement, but at
// most a quarter of the thumbnail: zoomed in, a small movement is a big part of the view.
const GATE_MS = 100, THUMB_W = 40, MIN_THUMB = 16, SEARCH = 6, SETTLE = 2;
const FAST_PX = 2, STEADY_PX = 0.8, CHANGE_PX = 3; // 1× thumbnail px (per sample for the first two)
const thumbCanvas = document.createElement("canvas");
const thumbCtx = thumbCanvas.getContext("2d", { willReadFrequently: true });
const gate = { state: "moving", fast: true, still: 0, noise: 0.1, prev: null, read: null, motion: null, drift: null, timer: 0 };

function thumbnail() {
  const { sx, sy, sw, sh } = visibleRect(video.videoWidth, video.videoHeight, "cover", dzoom);
  const w = Math.max(MIN_THUMB, Math.round(THUMB_W / zoom)), h = Math.max(16, Math.round((w * sh) / sw));
  // Draw at 4× and box-filter down: shrinking 1080p straight to 36 px aliases into fake motion.
  thumbCanvas.width = w * 4;
  thumbCanvas.height = h * 4;
  thumbCtx.imageSmoothingQuality = "high";
  thumbCtx.drawImage(video, sx, sy, sw, sh, 0, 0, w * 4, h * 4);
  const px = thumbCtx.getImageData(0, 0, w * 4, h * 4).data;
  const g = new Float32Array(w * h);
  for (let i = 0, y = 0; y < h * 4; y++) {
    for (let x = 0; x < w * 4; x++, i += 4) g[(y >> 2) * w + (x >> 2)] += (px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114) / 16;
  }
  const mean = g.reduce((a, v) => a + v, 0) / g.length;
  const sd = Math.max(6, Math.sqrt(g.reduce((a, v) => a + (v - mean) ** 2, 0) / g.length)); // floor: a blank wall is noise
  for (let i = 0; i < g.length; i++) g[i] = (g[i] - mean) / sd;
  return { g, w, h, u: THUMB_W / (w * zoom) }; // u: 1× thumbnail px per px of this one
}

const reach = (t) => Math.min(t.w >> 2, Math.max(2, Math.round(SEARCH / t.u))); // shift search, ±px

// Shift (dx, dy) of b's content against a (px of these thumbnails), within ±reach and refined to
// sub-pixel by a parabola fit, plus the mean abs difference left at that shift. The difference is
// taken at the sub-pixel shift (b sampled bilinearly): at the nearest whole pixel, a half-pixel
// offset of big text on a coarse thumbnail leaves as much as a changed scene would.
function compare(a, b) {
  if (!a || !b || a.w !== b.w || a.h !== b.h || a.u !== b.u) return null;
  const { w, h } = a, R = reach(a), n = 2 * R + 1;
  const cost = new Float32Array(n * n);
  let best = 0;
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      let s = 0;
      for (let y = R; y < h - R; y++) for (let x = R; x < w - R; x++) s += Math.abs(b.g[(y + dy) * w + x + dx] - a.g[y * w + x]);
      const k = (dy + R) * n + dx + R;
      cost[k] = s / ((w - 2 * R) * (h - 2 * R));
      if (cost[k] < cost[best]) best = k;
    }
  }
  const bx = best % n, by = (best / n) | 0;
  const vertex = (m, c, p) => (m - p) / (2 * (m - 2 * c + p) || 1);
  const fx = bx > 0 && bx < n - 1 ? vertex(cost[best - 1], cost[best], cost[best + 1]) : 0;
  const fy = by > 0 && by < n - 1 ? vertex(cost[best - n], cost[best], cost[best + n]) : 0;
  const dx = bx - R + fx, dy = by - R + fy, ix = Math.floor(dx), iy = Math.floor(dy), tx = dx - ix, ty = dy - iy;
  const at = (x, y) => b.g[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  let s = 0;
  for (let y = R; y < h - R; y++) {
    for (let x = R; x < w - R; x++) {
      const x0 = x + ix, y0 = y + iy;
      const top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * tx, bot = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * tx;
      s += Math.abs(top + (bot - top) * ty - a.g[y * w + x]);
    }
  }
  return { dx, dy, err: s / ((w - 2 * R) * (h - 2 * R)) };
}

function sample() {
  if (video.readyState < 2 || !video.videoWidth) return;
  const cur = thumbnail();
  const m = (gate.motion = compare(gate.prev, cur));
  gate.prev = cur;
  const same = (c) => c && c.err < 2 * gate.noise + 0.1;
  const dist = (c) => Math.hypot(c.dx, c.dy) * cur.u; // 1× thumbnail px
  const speed = m ? dist(m) : Infinity;
  const fast = (gate.fast = !same(m) || speed > FAST_PX);
  const steady = !fast && speed < STEADY_PX;
  if (steady) gate.noise = Math.min(0.3, Math.max(0.03, gate.noise + 0.2 * (m.err - gate.noise)));
  gate.still = steady ? gate.still + 1 : 0;
  const d = (gate.drift = compare(gate.read, cur));
  const keep = same(d); // still the scene the overlay was read from, maybe shifted
  gate.state = gate.still < SETTLE ? "moving" : !keep || dist(d) > CHANGE_PX ? "ready" : "idle";
  const show = !fast && keep;
  overlay.classList.toggle("faded", !show);
  if (show) {
    const k = view.clientWidth / cur.w;
    overlay.style.transform = `translate(${d.dx * k}px, ${d.dy * k}px)`;
  }
  stats.gate = gate.state;
  stats.fast = fast;
  if (DEBUG) debugReadout();
}

function startGate() {
  clearInterval(gate.timer);
  Object.assign(gate, { state: "moving", fast: true, still: 0, noise: 0.1, prev: null, read: null, motion: null, drift: null });
  overlay.style.transform = "";
  gate.timer = setInterval(sample, GATE_MS);
}

function stopGate() {
  clearInterval(gate.timer);
  overlay.classList.remove("faded");
  overlay.style.transform = "";
}

// ---------- drawing ----------
const octx = overlay.getContext("2d");

// Fit the canvas to the view and clear it (resizing only when the size changed: a frozen frame is
// repainted on every frame of a pinch).
function sizeCanvas(c) {
  const dpr = devicePixelRatio || 1;
  const w = Math.round(view.clientWidth * dpr), h = Math.round(view.clientHeight * dpr);
  const ctx = c.getContext("2d");
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
  } else {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, w, h);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function labelFor(unit, l) {
  if (l === "ja") return unit.roman && unit.roman !== unit.parts.map((p) => p.s).join("") ? unit.roman : "";
  return unit.parts.some((p) => p.rt) ? unit.roman : "";
}

const font = (px) => `600 ${px}px system-ui, -apple-system, sans-serif`;
const LABEL_GAP = 0.35; // em between neighbouring labels in a row

// Centres x for labels of widths w that want to sit at t (in order along the line), as close to t as
// possible without overlapping: least squares under x[i+1] - x[i] >= (w[i] + w[i+1]) / 2 + gap,
// which is an isotonic regression of t minus the minimum offsets (pool adjacent violators).
function spread(t, w, gap) {
  const o = [0];
  for (let i = 1; i < t.length; i++) o[i] = o[i - 1] + (w[i - 1] + w[i]) / 2 + gap;
  const blocks = [];
  t.forEach((ti, i) => {
    blocks.push({ sum: ti - o[i], n: 1 });
    while (blocks.length > 1 && blocks.at(-2).sum / blocks.at(-2).n >= blocks.at(-1).sum / blocks.at(-1).n) {
      const b = blocks.pop();
      blocks.at(-1).sum += b.sum;
      blocks.at(-1).n += b.n;
    }
  });
  const x = [];
  for (const b of blocks) for (let k = 0; k < b.n; k++) x.push(b.sum / b.n + o[x.length]);
  return x;
}

// Do two convex polygons (screen points) overlap by more than a pixel? Separating axis test.
function overlaps(p, q) {
  for (const poly of [p, q]) {
    for (let i = 0; i < poly.length; i++) {
      const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
      const nx = y1 - y2, ny = x2 - x1, n = Math.hypot(nx, ny) || 1;
      const a = p.map(([x, y]) => (x * nx + y * ny) / n), b = q.map(([x, y]) => (x * nx + y * ny) / n);
      if (Math.max(...a) < Math.min(...b) + 1 || Math.max(...b) < Math.min(...a) + 1) return false;
    }
  }
  return true;
}

// Rough overlap area (of the bounding boxes), to pick the least bad spot when no spot is clear.
function overlapArea(p, q) {
  const x = (pts) => pts.map((v) => v[0]), y = (pts) => pts.map((v) => v[1]);
  const w = Math.min(Math.max(...x(p)), Math.max(...x(q))) - Math.max(Math.min(...x(p)), Math.min(...x(q)));
  const h = Math.min(Math.max(...y(p)), Math.max(...y(q))) - Math.max(Math.min(...y(p)), Math.min(...y(q)));
  return Math.max(0, w) * Math.max(0, h);
}

// Each horizontal line gets one row of labels, its font scaled to that line (big text, big labels)
// and shrunk until the row fits the line's length; labels are then spread so none overlap. The row
// goes above the line, or below if that is clear of other lines and rows, or shrinks to fit the
// gap. Big text shrinks to 3/4 at most and then gets its row laid over its own characters (tightly
// stacked headlines leave no gap, and big glyphs stay readable under a label); small text shrinks
// to 1/4 (a dense paragraph's rows fit its thin gaps, readable once zoomed in) and then goes on the
// side that covers least. Vertical lines get their labels to the right of each character, like
// furigana.
// layout() works out where everything goes, at 1× (the screen mapping `map`), as a plan; paint()
// draws a plan through a transform. A frozen frame keeps its plan and repaints it on every frame of a
// pinch or drag, so zooming enlarges boxes and labels like a page, without rearranging them.
function layout(result, map) {
  const P = (x, y) => [map.ox + x * map.s, map.oy + y * map.s];
  const texts = result.lines.map((line) => line.box.map(([x, y]) => P(x, y)));
  const plan = { skipped: result.skipped.map((box) => box.map(([x, y]) => P(x, y))), boxes: texts, groups: [] };
  const rows = []; // label rows placed so far (screen polygons)
  const drawn = (plan.stats = { lines: 0, labels: 0, crowded: 0, over: 0, below: 0, overlaps: 0, collisions: 0, min: Infinity, max: 0, bigShrink: 1 });
  const vw = view.clientWidth, vh = view.clientHeight;
  // Biggest text first: its row is the hardest to place. Lines within 15 % of the same size go top to
  // bottom, so in a paragraph each row takes the space above its own line.
  const thickness = (k) => Math.hypot(...result.lines[k].B);
  const top = (k) => Math.min(...texts[k].map((p) => p[1]));
  const bySize = result.lines.map((_, k) => k).sort((a, b) => thickness(b) - thickness(a));
  const order = [];
  for (let i = 0, j; i < bySize.length; i = j) {
    for (j = i + 1; j < bySize.length && thickness(bySize[j]) * 1.15 > thickness(bySize[i]); j++);
    order.push(...bySize.slice(i, j).sort((a, b) => top(a) - top(b)));
  }
  for (const k of order) {
    const line = result.lines[k], note = result.notes[k];
    if (!note) continue;
    const xs = texts[k].map((p) => p[0]), ys = texts[k].map((p) => p[1]);
    if (Math.max(...xs) < 0 || Math.min(...xs) > vw || Math.max(...ys) < 0 || Math.min(...ys) > vh) continue; // off screen
    const labels = note.units
      .map((u) => ({ text: labelFor(u, result.lang), cs: line.chars.slice(u.i, u.i + u.n) }))
      .filter((l) => l.text && l.cs.length);
    if (!labels.length) continue;
    const bl = Math.hypot(...line.B) || 1, al = Math.hypot(...line.A) || 1;
    const bn = [line.B[0] / bl, line.B[1] / bl], an = [line.A[0] / al, line.A[1] / al]; // across, along
    const thick = bl * map.s, len = al * map.s;
    const base = Math.min(MAX_LABEL, thick * LABEL_SCALE);
    octx.font = font(100);
    const w100 = labels.map((l) => octx.measureText(l.text).width);
    const centres = labels.map((l) => P(l.cs.reduce((a, c) => a + c.x, 0) / l.cs.length, l.cs.reduce((a, c) => a + c.y, 0) / l.cs.length));
    drawn.lines++;

    if (line.vertical) {
      const size = base;
      plan.groups.push({
        line: k, size, over: false,
        rects: labels.map((l, i) => {
          const w = (w100[i] * size) / 100 + 6;
          return { cx: centres[i][0] + thick / 2 + 1 + w / 2, cy: centres[i][1], angle: 0, w, h: size * 1.3 };
        }),
        items: labels.map((l, i) => ({ text: l.text, x: centres[i][0] + thick / 2 + 4, y: centres[i][1], angle: 0, align: "left" })),
      });
      Object.assign(drawn, { labels: drawn.labels + labels.length, min: Math.min(drawn.min, size), max: Math.max(drawn.max, size) });
      continue;
    }

    // Row font: the line's own size, shrunk so all labels fit along the line (plus half a thickness
    // at each end) and across the screen.
    const [cx, cy] = P((line.box[0][0] + line.box[2][0]) / 2, (line.box[0][1] + line.box[2][1]) / 2);
    const t = centres.map(([x, y]) => (x - cx) * an[0] + (y - cy) * an[1]);
    const need = w100.reduce((a, b) => a + b, 0) / 100 + LABEL_GAP * (labels.length - 1);
    const fit = Math.min(base, Math.min(len + thick, vw - 8) / need);
    const place = (size, side) => {
      const w = w100.map((v) => (v * size) / 100);
      let x = spread(t, w, LABEL_GAP * size);
      let lo = Math.min(...x.map((xi, i) => xi - w[i] / 2)) - 4, hi = Math.max(...x.map((xi, i) => xi + w[i] / 2)) + 4;
      // Keep the row on screen when the line runs off an edge but is mostly on screen (rows of
      // near-horizontal lines only).
      if (Math.abs(an[0]) > 0.7 && cx >= 0 && cx <= vw) {
        const sx0 = cx + an[0] * (an[0] > 0 ? lo : hi), sx1 = cx + an[0] * (an[0] > 0 ? hi : lo);
        const d = (sx0 < 2 ? 2 - sx0 : sx1 > vw - 2 ? vw - 2 - sx1 : 0) / an[0];
        x = x.map((xi) => xi + d);
        lo += d;
        hi += d;
      }
      const off = side * (thick / 2 + size * 0.75); // along bn: negative is above the line
      const at = (u, v) => [cx + an[0] * u + bn[0] * (off + v), cy + an[1] * u + bn[1] * (off + v)];
      const poly = [at(lo, -size * 0.7), at(hi, -size * 0.7), at(hi, size * 0.7), at(lo, size * 0.7)];
      const others = texts.filter((_, j) => j !== k);
      const clear = !others.some((b) => overlaps(poly, b)) && !rows.some((r) => overlaps(poly, r));
      // Covering text is worse than covering another row of labels.
      const cost = () => others.reduce((a, b) => a + 2 * overlapArea(poly, b), 0) + rows.reduce((a, r) => a + overlapArea(poly, r), 0);
      return { size, side, w, x, lo, hi, at, poly, clear, cost };
    };
    const big = fit >= OVERLAY_MIN;
    const floor = fit * (big ? 0.75 : 0.25);
    // Each side's biggest clear row, shrinking from fit to floor. Above wins unless below is bigger;
    // for small text, unless it's much bigger: in a paragraph, a fraction of a pixel of difference
    // between the gaps mustn't flip rows to the other side, where they'd look like the next line's.
    const largest = (side) => {
      for (let size = fit; size >= floor; size = size > floor && size * 0.85 < floor ? floor : size * 0.85) {
        const r = place(size, side);
        if (r.clear) return r;
      }
      return null;
    };
    const up = largest(-1), down = largest(1);
    let row = up && !(down && down.size * (big ? 1 : 0.7) > up.size) ? up : down;
    if (!row && big) {
      // Centred on the line itself, if that doesn't land on another row; else the least bad spot.
      const over = { ...place(fit, 0), over: true };
      row = rows.some((r) => overlaps(over.poly, r)) ? [over, place(floor, -1), place(floor, 1)].sort((a, b) => a.cost() - b.cost())[0] : over;
      if (row.over) drawn.over++;
      else drawn.crowded++;
    }
    if (!row) {
      row = [place(floor, -1), place(floor, 1)].sort((a, b) => a.cost() - b.cost())[0];
      drawn.crowded++;
    }
    if (big) drawn.bigShrink = Math.min(drawn.bigShrink, row.size / fit);
    if (row.side > 0) drawn.below++;
    if (rows.some((r) => overlaps(row.poly, r))) drawn.collisions++;
    rows.push(row.poly);

    const angle = Math.atan2(an[1], an[0]);
    const [bx, by] = row.at((row.lo + row.hi) / 2, 0);
    plan.groups.push({
      line: k,
      size: row.size,
      over: !!row.over, // over the text: no backing, so the characters show between the letters; outlined labels instead
      rects: row.over ? [] : [{ cx: bx, cy: by, angle, w: row.hi - row.lo, h: row.size * 1.4 }],
      items: labels.map((l, i) => {
        const [x, y] = row.at(row.x[i], 0);
        return { text: l.text, x, y, angle, align: "center" };
      }),
    });
    for (let i = 1; i < labels.length; i++) if (row.x[i] - row.x[i - 1] < (row.w[i] + row.w[i - 1]) / 2 - 0.5) drawn.overlaps++;
    Object.assign(drawn, { labels: drawn.labels + labels.length, min: Math.min(drawn.min, row.size), max: Math.max(drawn.max, row.size) });
  }
  return plan;
}

function paint(plan, scale = { s: 1, tx: 0, ty: 0 }) {
  sizeCanvas(overlay);
  octx.transform(scale.s, 0, 0, scale.s, scale.tx, scale.ty);
  const outline = (pts) => {
    octx.beginPath();
    pts.forEach(([x, y], i) => (i ? octx.lineTo(x, y) : octx.moveTo(x, y)));
    octx.closePath();
    octx.stroke();
  };
  // How much of a row shows at this zoom: none below FADE_FROM px on screen, all of it from LEGIBLE.
  const shown = (g) => Math.min(1, Math.max(0, (g.size * scale.s - FADE_FROM) / (LEGIBLE - FADE_FROM)));
  const small = new Set(plan.groups.filter((g) => shown(g) < 1).map((g) => g.line));
  // Lines too small to read (live mode skips them) or to label at this zoom: a faint dashed hint to
  // move closer or zoom in. Outlines stay as thin on screen at any zoom.
  octx.setLineDash([4 / scale.s, 4 / scale.s]);
  octx.lineWidth = 1 / scale.s;
  octx.strokeStyle = "rgba(255,255,255,0.4)";
  plan.skipped.forEach(outline);
  plan.boxes.forEach((b, k) => small.has(k) && outline(b));
  octx.setLineDash([]);
  octx.lineWidth = 1.5 / scale.s;
  octx.strokeStyle = "rgba(255,255,255,0.75)";
  plan.boxes.forEach((b, k) => small.has(k) || outline(b));
  octx.textBaseline = "middle";
  octx.lineJoin = "round";
  for (const g of plan.groups) {
    if (!(octx.globalAlpha = shown(g))) continue;
    octx.fillStyle = "rgba(0,0,0,0.6)";
    for (const r of g.rects) {
      octx.save();
      octx.translate(r.cx, r.cy);
      octx.rotate(r.angle);
      octx.fillRect(-r.w / 2, -r.h / 2, r.w, r.h);
      octx.restore();
    }
    octx.font = font(g.size);
    octx.fillStyle = "#ffe066";
    octx.strokeStyle = "rgba(0,0,0,0.85)";
    octx.lineWidth = Math.max(2, g.size * 0.16);
    for (const t of g.items) {
      octx.save();
      octx.translate(t.x, t.y);
      octx.rotate(t.angle);
      octx.textAlign = t.align;
      if (g.over) octx.strokeText(t.text, 0, 0);
      octx.fillText(t.text, 0, 0);
      octx.restore();
    }
  }
  octx.globalAlpha = 1;
  // hidden: rows not fully shown at this zoom.
  stats.labels = { ...plan.stats, hidden: small.size, min: plan.stats.min * scale.s, max: plan.stats.max * scale.s }; // on-screen px
}

// The live overlay: laid out and painted at once. The plan is kept for the shutter, which shows these
// labels on the frozen frame until its own read is done.
let livePlan = null;
function draw(result, map) {
  livePlan = layout(result, map);
  paint(livePlan);
}

// Fill the list of lines with ruby text; the list button (#list) shows it.
function fillSheet(result) {
  linesEl.replaceChildren(...result.notes.map((note) => {
    const li = document.createElement("li");
    const text = document.createElement("div");
    text.className = "ruby";
    text.lang = result.lang;
    for (const u of note.units) {
      for (const p of u.parts) {
        if (!p.rt) { text.append(p.s); continue; }
        const ruby = document.createElement("ruby");
        const rt = document.createElement("rt");
        rt.textContent = p.rt;
        ruby.append(p.s, rt);
        text.append(ruby);
      }
    }
    const roman = document.createElement("div");
    roman.className = "roman";
    roman.textContent = note.roman;
    li.append(text, roman);
    return li;
  }));
  if (!result.notes.length) linesEl.innerHTML = "<li class=empty>No text found. Try moving closer or holding steady.</li>";
}

function toggleSheet(open = sheet.hidden) {
  sheet.hidden = !open;
  $("#list").setAttribute("aria-pressed", String(open));
}

function setStatus(msg, error = false) {
  if (msg instanceof Error) {
    console.error(msg);
    msg = msg.where ? `${msg.message} (${msg.where})` : msg.message;
  }
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", error);
}
const timing = (r) => `det ${r.ms.det.toFixed(0)} ms · rec ${r.ms.rec.toFixed(0)} ms · ${r.lines.length} lines` +
  `${r.skipped.length ? ` (+${r.skipped.length} small)` : ""} · ${backend}×${THREADS} ${PRECISION}`;

// ---------- dev panel ----------
// Tap the status line three times: gate and viewport numbers, plus backend/threads/spin switches that
// reload the app. For tuning on a phone, where a home-screen app takes no URL switches.
let probe = null;
function debugReadout() {
  if (!DEBUG) return;
  if (!probe) {
    probe = document.createElement("div");
    probe.style.cssText = "position:fixed;visibility:hidden;padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)";
    document.body.append(probe);
  }
  const s = getComputedStyle(probe);
  const t = gate.prev, u = t?.u ?? 1; // shifts in 1× thumbnail px, like the thresholds
  const f = (c) => (c ? `${(c.dx * u).toFixed(2)},${(c.dy * u).toFixed(2)} e${c.err.toFixed(2)}` : "-");
  $("#debug").textContent = [
    `gate ${gate.state}${gate.fast ? " fast" : ""} still ${gate.still} noise ${gate.noise.toFixed(2)} motion ${f(gate.motion)} drift ${f(gate.drift)}`,
    `reads ${stats.reads} busy ${(stats.busy / 1000).toFixed(1)} s · ${backend}×${THREADS} spin ${SPIN ? 1 : 0}${stats.backend.gpu ? ` · gpu ${stats.backend.gpu}` : ""}`,
    `zoom ${zoom.toFixed(2)}× (${hwZoom ? `camera ${hwZoom.min}–${hwZoom.max}×, digital ${dzoom.toFixed(2)}×` : "digital"})${t ? ` · thumb ${t.w}px ±${reach(t)} u${u.toFixed(2)}` : ""}${mode === "still" ? ` · still ${sv.s.toFixed(2)}×` : ""}`,
    `inner ${innerWidth}×${innerHeight} screen ${screen.width}×${screen.height} vv ${visualViewport?.height.toFixed(0)} view ${view.clientHeight}`,
    `safe ${[s.paddingTop, s.paddingRight, s.paddingBottom, s.paddingLeft].map(parseFloat).join("/")} standalone ${navigator.standalone ?? matchMedia("(display-mode: standalone)").matches}`,
    ...(stats.backend.fallback ? [stats.backend.fallback] : []),
  ].join("\n");
}

function saveDev(change) {
  Object.assign(dev, change);
  try { localStorage.setItem("dev", JSON.stringify(dev)); } catch {}
}

function showDev() {
  $("#debug").hidden = $("#devbar").hidden = !DEBUG;
  if (!DEBUG) return;
  const opt = (label, key, value, on) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.setAttribute("aria-pressed", String(on));
    b.onclick = () => { saveDev({ [key]: value }); location.reload(); };
    return b;
  };
  const threads = Number(dev.threads) || 0; // 0: automatic
  $("#devbar").replaceChildren(
    opt("WASM", "ep", "wasm", EP === "wasm"), opt("WebGPU", "ep", "webgpu", EP === "webgpu"),
    ...[0, 1, 2, 4].map((n) => opt(n ? `${n} thr` : "auto thr", "threads", n || null, threads === n)),
    ...[10, 14, 20].map((n) => opt(`line ≥${n}px`, "minline", n === 14 ? null : n, MIN_LINE === n)),
    opt(SPIN ? "spin on" : "spin off", "spin", SPIN ? 0 : 1, SPIN),
    opt(String(setting("zoomhw")) === "0" ? "digital zoom" : "camera zoom", "zoomhw", String(setting("zoomhw")) === "0" ? null : 0, false),
  );
  debugReadout();
}

let taps = [];
statusEl.addEventListener("click", () => {
  const now = performance.now();
  taps = [...taps.filter((t) => now - t < 700), now];
  if (taps.length < 3) return;
  taps = [];
  DEBUG = !DEBUG;
  saveDev({ debug: DEBUG });
  showDev();
});

// ---------- zoom and gestures ----------
// Live: pinch or the slider. With the camera's own zoom where the browser has it; otherwise digital,
// where the video is scaled up with CSS and each read crops the middle of the frame, so the OCR sees
// the zoomed text at the camera's full resolution (finer than the 1× read, up to ~2×).
// Still: pinch and drag the frozen frame; it is repainted on every animation frame while the fingers
// move (the image from its full-resolution source, the labels from the plan of its read).
// Double-tap resets either.
// The camera's own zoom goes to 5× (telephoto lenses top out there; beyond it the camera only crops),
// digital zoom to 3× (softer than that isn't worth it). A frozen frame zooms to 5×: its labels stay
// sharp at any size, the picture under them doesn't.
const DIGITAL_MAX = 3, CAMERA_MAX = 5, STILL_ZOOM_MAX = 5;
const zoomMax = () => (hwZoom ? Math.max(DIGITAL_MAX, Math.min(CAMERA_MAX, hwZoom.max)) : DIGITAL_MAX);

let tickMax = 0;
function applyZoom() {
  const hw = hwZoom ? Math.min(zoom, hwZoom.max) : 1;
  dzoom = zoom / hw;
  if (hwZoom) cameraZoom(Math.max(hwZoom.min, hw));
  video.style.transform = dzoom > 1.001 ? `scale(${dzoom})` : "";
  const max = zoomMax();
  if (max !== tickMax) {
    // Marks under the slider at 1×, 2×, 3× (and 5×), where the thumb's centre is at those values.
    tickMax = max;
    $("#zoom").max = String(max);
    $("#zoombar .ticks").replaceChildren(...[1, 2, 3, 5].filter((v) => v <= max).map((v) => {
      const b = document.createElement("button");
      b.textContent = `${v}×`;
      b.style.left = `calc(9px + (100% - 18px) * ${(v - 1) / (max - 1)})`;
      b.onclick = () => { setZoom(v); wakeZoombar(); };
      return b;
    }));
  }
  $("#zoom").value = String(zoom);
  $("#zoomval").textContent = `${zoom.toFixed(1)}×`;
  stats.zoom = { live: zoom, digital: dzoom, camera: !!hwZoom, still: sv.s, max };
}

// The slider rests as a thin line; touching it (or pinching) shows it in full for a couple of seconds.
let zoombarTimer = 0;
function wakeZoombar() {
  $("#zoombar").classList.add("active");
  clearTimeout(zoombarTimer);
  zoombarTimer = setTimeout(() => $("#zoombar").classList.remove("active"), 2000);
}

function setZoom(z) {
  z = Math.min(zoomMax(), Math.max(1, Math.round(z * 20) / 20));
  if (z === zoom) return;
  zoom = z;
  applyZoom();
  if (mode === "live") { gate.read = null; sizeCanvas(overlay); } // the labels belong to the old zoom
}

// applyConstraints is slow: keep one in flight and apply only the latest value after it.
let camZoomNext = null, camZoomBusy = false;
async function cameraZoom(v) {
  camZoomNext = v;
  if (camZoomBusy) return;
  camZoomBusy = true;
  while (camZoomNext != null) {
    const z = camZoomNext;
    camZoomNext = null;
    try { await stream?.getVideoTracks()[0]?.applyConstraints({ advanced: [{ zoom: z }] }); } catch {}
  }
  camZoomBusy = false;
}

const viewMap = (map) => ({ s: map.s * sv.s, ox: map.ox * sv.s + sv.tx, oy: map.oy * sv.s + sv.ty });

// The frozen frame, from its full-resolution source, at the current still zoom.
function paintStill() {
  const { src, rect, k, map } = lastStill, m = viewMap(map);
  const ctx = sizeCanvas(still);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, view.clientWidth, view.clientHeight);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, rect.sx, rect.sy, rect.sw, rect.sh, m.ox, m.oy, rect.sw * k * m.s, rect.sh * k * m.s);
}

// The frozen frame and its labels (its own read's, or the live ones until that is done).
function redrawStill() {
  if (!lastStill) return;
  paintStill();
  if (lastStill.plan) paint(lastStill.plan, sv);
  stats.zoom = { ...stats.zoom, still: sv.s, pan: [sv.tx, sv.ty] };
}

// During a pinch or drag: at most one repaint per animation frame.
let stillFrame = 0;
function setStillView(next) {
  const s = Math.min(STILL_ZOOM_MAX, Math.max(1, next.s));
  const vw = view.clientWidth, vh = view.clientHeight;
  sv = { s, tx: Math.min(0, Math.max(vw * (1 - s), next.tx)), ty: Math.min(0, Math.max(vh * (1 - s), next.ty)) };
  stillFrame ||= requestAnimationFrame(() => { stillFrame = 0; redrawStill(); });
}

const fingers = new Map(); // pointerId -> [x, y]
let gesture = null, multi = false, lastTap = 0;
const centroid = (pts) => [pts.reduce((a, p) => a + p[0], 0) / pts.length, pts.reduce((a, p) => a + p[1], 0) / pts.length];
const spanOf = (pts) => (pts.length > 1 ? Math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]) : 0);

function gestureFrom() {
  const pts = [...fingers.values()];
  gesture = { c: centroid(pts), d: spanOf(pts), zoom, sv: { ...sv }, moved: false };
}

view.addEventListener("pointerdown", (e) => {
  view.setPointerCapture(e.pointerId);
  fingers.set(e.pointerId, [e.clientX, e.clientY]);
  if (fingers.size > 1) multi = true;
  gestureFrom();
});

view.addEventListener("pointermove", (e) => {
  if (!fingers.has(e.pointerId) || !gesture) return;
  fingers.set(e.pointerId, [e.clientX, e.clientY]);
  const pts = [...fingers.values()], c = centroid(pts), d = spanOf(pts);
  if (Math.hypot(c[0] - gesture.c[0], c[1] - gesture.c[1]) > 8 || Math.abs(d - gesture.d) > 8) gesture.moved = true;
  if (!gesture.moved) return;
  const f = gesture.d && d ? d / gesture.d : 1;
  if (mode === "live" && pts.length > 1) { setZoom(gesture.zoom * f); wakeZoombar(); }
  if (mode === "still") {
    // The image point under the fingers' starting centre stays under their centre.
    const g = gesture.sv, s = Math.min(STILL_ZOOM_MAX, Math.max(1, g.s * f));
    setStillView({ s, tx: c[0] - (s * (gesture.c[0] - g.tx)) / g.s, ty: c[1] - (s * (gesture.c[1] - g.ty)) / g.s });
  }
});

function lift(e) {
  if (!fingers.delete(e.pointerId)) return;
  if (fingers.size) return gestureFrom(); // a finger is still down: carry on from here
  const tap = !multi && gesture && !gesture.moved;
  if (mode === "still" && !tap) redrawStill();
  if (tap) {
    const now = performance.now();
    if (now - lastTap < 350) {
      lastTap = 0;
      if (mode === "live") setZoom(1);
      else if (mode === "still") { sv = { s: 1, tx: 0, ty: 0 }; redrawStill(); }
    } else lastTap = now;
  }
  gesture = null;
  multi = false;
}
view.addEventListener("pointerup", lift);
view.addEventListener("pointercancel", lift);
document.addEventListener("gesturestart", (e) => e.preventDefault()); // Safari's own pinch-zoom of the page

$("#zoom").oninput = (e) => { setZoom(Number(e.target.value)); wakeZoombar(); };
$("#zoom").addEventListener("pointerdown", wakeZoombar);

// ---------- modes ----------
function setMode(m) {
  mode = stats.mode = document.body.dataset.mode = m;
}

function setShutter(label, text) {
  $("#shutter").setAttribute("aria-label", label);
  $("#shutter").textContent = text;
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  // The camera's own zoom where the browser exposes it; the dev panel can force digital zoom.
  const z = stream.getVideoTracks()[0]?.getCapabilities?.().zoom;
  hwZoom = z && z.max > 1 && String(setting("zoomhw")) !== "0" ? { min: z.min, max: z.max } : null;
  applyZoom();
}

function stopCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
}

async function live() {
  setMode("live");
  still.hidden = true;
  setShutter("Freeze", "");
  toggleSheet(false);
  sizeCanvas(overlay);
  if (!stream) await startCamera();
  else await video.play(); // resuming after a freeze, which only paused the video
  setStatus("Loading models…");
  startGate();
  liveLoop(++loopToken);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Reads only when the motion gate says the view is steady and new, and stays at most ~50 % busy.
async function liveLoop(token) {
  let next = 0;
  while (mode === "live" && token === loopToken) {
    await sleep(GATE_MS);
    if (gate.state !== "ready" || performance.now() < next || video.readyState < 2) continue;
    const at = thumbnail(); // of the frame being read: drift is measured from it
    const { image, map } = grab(video, video.videoWidth, video.videoHeight, LIVE_MAX, "cover", dzoom);
    const t0 = performance.now();
    try {
      const r = await read(image, LIVE_DET, MIN_LINE / map.s);
      if (mode !== "live" || token !== loopToken) break;
      const took = performance.now() - t0;
      stats.busy += took;
      next = performance.now() + Math.max(MIN_GAP, took);
      gate.read = at;
      overlay.style.transform = "";
      draw(r, map);
      sample(); // shift the new overlay to where the text is now, or hide it if the view moved meanwhile
      setStatus(timing(r));
    } catch (e) {
      if (mode !== "live" || token !== loopToken) break; // cancelled by the shutter, or stopped
      setStatus(e, true);
      await sleep(1000);
    }
  }
}

// Read a still image (frozen camera frame or picked photo) at higher resolution and list the lines.
// keep: the shutter froze the live view; its labels stay up until the new read is drawn.
async function readStill(src, w, h, fit, keep = false) {
  ocr({ type: "cancel" }).catch(() => {}); // drop the live read (or older still read) in progress
  keep &&= mode === "live" && !overlay.classList.contains("faded");
  setMode("still");
  loopToken++;
  stopGate();
  setShutter("Back to live camera", "Live");
  const { image, map, rect, k } = grab(src, w, h, STILL_MAX, fit);
  // The live labels (same screen mapping: the same part of the same frame) until this read is done.
  lastStill = { src, w, h, fit, map, rect, k, result: null, plan: keep ? livePlan : null };
  sv = { s: 1, tx: 0, ty: 0 };
  if (!keep) sizeCanvas(overlay);
  overlay.style.transform = "";
  redrawStill();
  still.hidden = false;
  setStatus("Reading…");
  try {
    const r = await read(image, STILL_DET);
    if (mode !== "still" || lastStill.src !== src) return;
    lastStill.result = r;
    lastStill.plan = layout(r, map);
    redrawStill(); // at whatever zoom the fingers have reached meanwhile
    fillSheet(r);
    setStatus(timing(r));
  } catch (e) {
    if (!e.cancelled) setStatus(e, true); // cancelled: a newer still read took over
  }
}

sheet.onclick = () => toggleSheet(false);
$("#list").onclick = () => toggleSheet();

let snapping = false;
$("#shutter").onclick = async () => {
  if (mode === "live" && video.videoWidth) {
    if (snapping) return;
    snapping = true;
    // Freeze what's on screen at the camera's full resolution, so it can be zoomed into.
    const r = visibleRect(video.videoWidth, video.videoHeight, "cover", dzoom);
    const bmp = await createImageBitmap(video, Math.round(r.sx), Math.round(r.sy), Math.round(r.sw), Math.round(r.sh)).finally(() => (snapping = false));
    video.pause();
    readStill(bmp, bmp.width, bmp.height, "cover", true);
  } else {
    live().catch((e) => setStatus(e, true));
  }
};

$("#file").onchange = async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  $("#start").hidden = true;
  const bmp = await createImageBitmap(file);
  stopCamera();
  readStill(bmp, bmp.width, bmp.height, "contain");
};

$("#go").onclick = async () => {
  try {
    await live();
    $("#start").hidden = true;
  } catch (e) {
    $("#start-msg").textContent = `Camera unavailable (${e.message}). You can still open a photo.`;
  }
};

function setLang(l) {
  lang = l;
  try { localStorage.setItem("lang", l); } catch {}
  document.querySelectorAll("[data-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === l)));
  ensureModels(l).then(() => l === lang && showReady(), (e) => setStatus(e, true));
  if (mode === "live") gate.read = null; // the overlay is in the old language: read again
  if (mode === "still" && lastStill) readStill(lastStill.src, lastStill.w, lastStill.h, lastStill.fit);
}
document.querySelectorAll("[data-lang]").forEach((b) => (b.onclick = () => setLang(b.dataset.lang)));

document.addEventListener("visibilitychange", () => {
  if (document.hidden) { stopCamera(); stopGate(); loopToken++; }
  else if (mode === "live") live().catch((e) => setStatus(e, true));
});
addEventListener("resize", () => { if (mode === "still" && lastStill) readStill(lastStill.src, lastStill.w, lastStill.h, lastStill.fit); });

if (!navigator.mediaDevices?.getUserMedia) $("#start-msg").textContent = "No camera access here (needs HTTPS). You can still open a photo.";
showDev();
setLang(lang);
inited.catch((e) => { if (!reloading) setStatus(`Model load failed: ${e.message}`, true); });
