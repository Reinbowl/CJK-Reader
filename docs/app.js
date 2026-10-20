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
// Labels: LABEL_SCALE × the line's thickness, between MIN_LABEL and MAX_LABEL px. Live mode doesn't
// recognise lines thinner than MIN_LINE px on screen (dev panel: 10/14/20).
const LABEL_SCALE = 0.45, MIN_LABEL = 8, MAX_LABEL = 28;
const MIN_LINE = Number(setting("minline")) || 14;
const MIN_GAP = 250; // live mode: ms between reads, at least, and at least as long as the last read took

// What the app is doing, for the e2e tests (test/e2e.test.mjs) and ?debug.
const stats = (self.cjkStats = { mode: "idle", backend: "", reads: 0, lines: 0, skipped: 0, recognised: 0, busy: 0, ms: null, gate: null, fast: null, labels: null, reading: 0, cancelled: 0 });

let lang = (() => {
  try { const saved = localStorage.getItem("lang"); if (MODEL[saved]) return saved; } catch {}
  const nav = (navigator.language || "").slice(0, 2);
  return MODEL[nav] ? nav : "zh";
})();
let mode = "idle"; // idle | live | still
let stream = null, loopToken = 0, lastStill = null;

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

const progress = {};
function onProgress({ type, file, loaded, total }) {
  if (type !== "progress") return;
  progress[file] = { loaded, total };
  const parts = Object.entries(progress).map(([f, p]) =>
    `${f.split(".")[0]} ${p.total && p.loaded >= p.total ? "✓" : `${(p.loaded / 1e6).toFixed(1)}${p.total ? `/${(p.total / 1e6).toFixed(1)}` : ""} MB`}`);
  $("#progress").textContent = parts.join(" · ");
}

// The OCR worker also says when it had to move from WebGPU to WASM mid-session.
const ocr = rpc(new Worker("ocr-worker.js", { type: "module" }), (d) => (d.type === "backend" ? onBackend(d) : onProgress(d)));
const reading = rpc(new Worker("reading-worker.js"), onProgress);
const inited = ocr({ type: "init", ep: EP, precision: PRECISION, threads: THREADS, spin: SPIN });
let backend = EP;
function onBackend(r) {
  backend = r.fallback ? `${r.backend} (no webgpu)` : r.backend;
  stats.backend = { ...r };
  if (r.fallback) console.warn(r.fallback);
  debugReadout();
}
inited.then(onBackend, () => {});
const loaded = {};
function ensureModels(l) {
  loaded[l] ??= Promise.all([inited, ocr({ type: "load", model: MODEL[l] }), reading({ type: "load", lang: l })])
    .catch((e) => { delete loaded[l]; throw e; });
  return loaded[l];
}

// ---------- capture ----------
const grabCanvas = document.createElement("canvas");
const grabCtx = grabCanvas.getContext("2d", { willReadFrequently: true });

// The part of a srcW×srcH source that is on screen. fit "cover" matches the video's CSS (centre
// crop); "contain" shows a whole photo.
function visibleRect(srcW, srcH, fit) {
  if (fit !== "cover") return { sx: 0, sy: 0, sw: srcW, sh: srcH };
  const s = Math.max(view.clientWidth / srcW, view.clientHeight / srcH);
  const sw = view.clientWidth / s, sh = view.clientHeight / s;
  return { sx: (srcW - sw) / 2, sy: (srcH - sh) / 2, sw, sh };
}

// Draw the part of `src` that is on screen into grabCanvas (at most maxSide px) and return its
// pixels plus the image->screen mapping.
function grab(src, srcW, srcH, maxSide, fit) {
  const vw = view.clientWidth, vh = view.clientHeight;
  const { sx, sy, sw, sh } = visibleRect(srcW, srcH, fit);
  const k = Math.min(1, maxSide / Math.max(sw, sh));
  const w = Math.round(sw * k), h = Math.round(sh * k);
  grabCanvas.width = w; grabCanvas.height = h;
  grabCtx.drawImage(src, sx, sy, sw, sh, 0, 0, w, h);
  const s = Math.min(vw / w, vh / h);
  // A whole photo sits just under the header so the results sheet covers less of it.
  const oy = fit === "contain" ? Math.min((vh - h * s) / 2, $("header").offsetHeight) : (vh - h * s) / 2;
  return { image: grabCtx.getImageData(0, 0, w, h), map: { s, ox: (vw - w * s) / 2, oy } };
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
const GATE_MS = 100, THUMB_W = 40, SEARCH = 6, SETTLE = 2;
const FAST_PX = 2, STEADY_PX = 0.8, CHANGE_PX = 3; // thumbnail px (per sample for the first two)
const thumbCanvas = document.createElement("canvas");
const thumbCtx = thumbCanvas.getContext("2d", { willReadFrequently: true });
const gate = { state: "moving", fast: true, still: 0, noise: 0.1, prev: null, read: null, motion: null, drift: null, timer: 0 };

function thumbnail() {
  const { sx, sy, sw, sh } = visibleRect(video.videoWidth, video.videoHeight, "cover");
  const w = THUMB_W, h = Math.max(2 * SEARCH + 2, Math.round((THUMB_W * sh) / sw));
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
  return { g, w, h };
}

// Shift (dx, dy) of b's content against a, within ±SEARCH px and refined to sub-pixel by a parabola
// fit, plus the mean abs difference left at that shift.
function compare(a, b) {
  if (!a || !b || a.w !== b.w || a.h !== b.h) return null;
  const { w, h } = a, R = SEARCH, n = 2 * R + 1;
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
  return { dx: bx - R + fx, dy: by - R + fy, err: cost[best] };
}

function sample() {
  if (video.readyState < 2 || !video.videoWidth) return;
  const cur = thumbnail();
  const m = (gate.motion = compare(gate.prev, cur));
  gate.prev = cur;
  const same = (c) => c && c.err < 2 * gate.noise + 0.1;
  const speed = m ? Math.hypot(m.dx, m.dy) : Infinity;
  const fast = (gate.fast = !same(m) || speed > FAST_PX);
  const steady = !fast && speed < STEADY_PX;
  if (steady) gate.noise = Math.min(0.3, Math.max(0.03, gate.noise + 0.2 * (m.err - gate.noise)));
  gate.still = steady ? gate.still + 1 : 0;
  const d = (gate.drift = compare(gate.read, cur));
  const keep = same(d); // still the scene the overlay was read from, maybe shifted
  gate.state = gate.still < SETTLE ? "moving" : !keep || Math.hypot(d.dx, d.dy) > CHANGE_PX ? "ready" : "idle";
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

// keep: leave the overlay where it is (the shutter keeps the live labels up while re-reading).
function stopGate(keep = false) {
  clearInterval(gate.timer);
  overlay.classList.remove("faded");
  if (!keep) overlay.style.transform = "";
}

// ---------- drawing ----------
const octx = overlay.getContext("2d");

function sizeCanvas(c) {
  const dpr = devicePixelRatio || 1;
  c.width = Math.round(view.clientWidth * dpr);
  c.height = Math.round(view.clientHeight * dpr);
  const ctx = c.getContext("2d");
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
// gap; if nothing fits it goes on the side that covers least, so every reading is shown. Vertical
// lines get their labels to the right of each character, like furigana.
function draw(result, map) {
  sizeCanvas(overlay);
  const P = (x, y) => [map.ox + x * map.s, map.oy + y * map.s];
  const outline = (box) => {
    octx.beginPath();
    box.forEach(([x, y], i) => (i ? octx.lineTo : octx.moveTo).call(octx, ...P(x, y)));
    octx.closePath();
    octx.stroke();
  };
  // Lines too small to label (live mode doesn't read them): a faint dashed hint to move closer.
  octx.setLineDash([4, 4]);
  octx.lineWidth = 1;
  octx.strokeStyle = "rgba(255,255,255,0.4)";
  result.skipped.forEach(outline);
  octx.setLineDash([]);
  octx.lineWidth = 1.5;
  octx.strokeStyle = "rgba(255,255,255,0.75)";
  result.lines.forEach((line) => outline(line.box));

  const texts = result.lines.map((line) => line.box.map(([x, y]) => P(x, y)));
  const rows = []; // label rows placed so far (screen polygons)
  const drawn = { lines: 0, labels: 0, crowded: 0, overlaps: 0, min: Infinity, max: 0 };
  octx.textBaseline = "middle";
  // Biggest text first: its row is the hardest to place.
  const order = result.lines.map((_, k) => k).sort((a, b) => Math.hypot(...result.lines[b].B) - Math.hypot(...result.lines[a].B));
  for (const k of order) {
    const line = result.lines[k], note = result.notes[k];
    if (!note) continue;
    const labels = note.units
      .map((u) => ({ text: labelFor(u, result.lang), cs: line.chars.slice(u.i, u.i + u.n) }))
      .filter((l) => l.text && l.cs.length);
    if (!labels.length) continue;
    const bl = Math.hypot(...line.B) || 1, al = Math.hypot(...line.A) || 1;
    const bn = [line.B[0] / bl, line.B[1] / bl], an = [line.A[0] / al, line.A[1] / al]; // across, along
    const thick = bl * map.s, len = al * map.s;
    const base = Math.min(MAX_LABEL, Math.max(MIN_LABEL, thick * LABEL_SCALE));
    octx.font = font(100);
    const w100 = labels.map((l) => octx.measureText(l.text).width);
    const centres = labels.map((l) => P(l.cs.reduce((a, c) => a + c.x, 0) / l.cs.length, l.cs.reduce((a, c) => a + c.y, 0) / l.cs.length));
    drawn.lines++;

    if (line.vertical) {
      const size = base;
      octx.font = font(size);
      for (const [i, l] of labels.entries()) {
        const [x, y] = centres[i];
        const w = (w100[i] * size) / 100;
        octx.fillStyle = "rgba(0,0,0,0.6)";
        octx.fillRect(x + thick / 2 + 1, y - size * 0.65, w + 6, size * 1.3);
        octx.fillStyle = "#ffe066";
        octx.textAlign = "left";
        octx.fillText(l.text, x + thick / 2 + 4, y);
      }
      Object.assign(drawn, { labels: drawn.labels + labels.length, min: Math.min(drawn.min, size), max: Math.max(drawn.max, size) });
      continue;
    }

    // Row font: the line's own size, shrunk so all labels fit along the line (plus half a thickness
    // at each end) and across the screen; never below MIN_LABEL, where the row may run past the line.
    const [cx, cy] = P((line.box[0][0] + line.box[2][0]) / 2, (line.box[0][1] + line.box[2][1]) / 2);
    const t = centres.map(([x, y]) => (x - cx) * an[0] + (y - cy) * an[1]);
    const need = w100.reduce((a, b) => a + b, 0) / 100 + LABEL_GAP * (labels.length - 1);
    const fit = Math.max(MIN_LABEL, Math.min(base, Math.min(len + thick, view.clientWidth - 8) / need));
    const layout = (size, side) => {
      const w = w100.map((v) => (v * size) / 100);
      let x = spread(t, w, LABEL_GAP * size);
      let lo = Math.min(...x.map((xi, i) => xi - w[i] / 2)) - 4, hi = Math.max(...x.map((xi, i) => xi + w[i] / 2)) + 4;
      // Keep the row on screen when the line runs off an edge (rows of near-horizontal lines only).
      if (Math.abs(an[0]) > 0.7) {
        const sx0 = cx + an[0] * (an[0] > 0 ? lo : hi), sx1 = cx + an[0] * (an[0] > 0 ? hi : lo);
        const d = (sx0 < 2 ? 2 - sx0 : sx1 > view.clientWidth - 2 ? view.clientWidth - 2 - sx1 : 0) / an[0];
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
      return { size, w, x, lo, hi, at, poly, clear, cost };
    };
    let row = null;
    for (let size = fit; !row && size >= MIN_LABEL; size = size > MIN_LABEL && size * 0.85 < MIN_LABEL ? MIN_LABEL : size * 0.85) {
      row = [layout(size, -1), layout(size, 1)].find((r) => r.clear);
    }
    if (!row) {
      row = [layout(MIN_LABEL, -1), layout(MIN_LABEL, 1)].sort((a, b) => a.cost() - b.cost())[0];
      drawn.crowded++;
    }
    rows.push(row.poly);

    const angle = Math.atan2(an[1], an[0]);
    const [bx, by] = row.at((row.lo + row.hi) / 2, 0);
    octx.save();
    octx.translate(bx, by);
    octx.rotate(angle);
    octx.fillStyle = "rgba(0,0,0,0.6)";
    octx.fillRect(-(row.hi - row.lo) / 2, -row.size * 0.7, row.hi - row.lo, row.size * 1.4);
    octx.restore();
    octx.font = font(row.size);
    octx.textAlign = "center";
    octx.fillStyle = "#ffe066";
    for (const [i, l] of labels.entries()) {
      const [x, y] = row.at(row.x[i], 0);
      octx.save();
      octx.translate(x, y);
      octx.rotate(angle);
      octx.fillText(l.text, 0, 0);
      octx.restore();
      if (i && row.x[i] - row.x[i - 1] < (row.w[i] + row.w[i - 1]) / 2 - 0.5) drawn.overlaps++;
    }
    Object.assign(drawn, { labels: drawn.labels + labels.length, min: Math.min(drawn.min, row.size), max: Math.max(drawn.max, row.size) });
  }
  stats.labels = drawn;
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
  const f = (c) => (c ? `${c.dx.toFixed(2)},${c.dy.toFixed(2)} e${c.err.toFixed(2)}` : "-");
  $("#debug").textContent = [
    `gate ${gate.state}${gate.fast ? " fast" : ""} still ${gate.still} noise ${gate.noise.toFixed(2)} motion ${f(gate.motion)} drift ${f(gate.drift)}`,
    `reads ${stats.reads} busy ${(stats.busy / 1000).toFixed(1)} s · ${backend}×${THREADS} spin ${SPIN ? 1 : 0}${stats.backend.gpu ? ` · gpu ${stats.backend.gpu}` : ""}`,
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
    const { image, map } = grab(video, video.videoWidth, video.videoHeight, LIVE_MAX, "cover");
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
  stopGate(keep);
  setShutter("Back to live camera", "Live");
  const { image, map } = grab(src, w, h, STILL_MAX, fit);
  const sctx = sizeCanvas(still);
  sctx.fillStyle = "#000";
  sctx.fillRect(0, 0, view.clientWidth, view.clientHeight);
  sctx.drawImage(grabCanvas, map.ox, map.oy, image.width * map.s, image.height * map.s);
  still.hidden = false;
  lastStill = { src, w, h, fit };
  if (!keep) sizeCanvas(overlay);
  setStatus("Reading…");
  try {
    const r = await read(image, STILL_DET);
    if (mode !== "still") return;
    overlay.style.transform = "";
    draw(r, map);
    fillSheet(r);
    setStatus(timing(r));
  } catch (e) {
    if (!e.cancelled) setStatus(e, true); // cancelled: a newer still read took over
  }
}

sheet.onclick = () => toggleSheet(false);
$("#list").onclick = () => toggleSheet();

$("#shutter").onclick = () => {
  if (mode === "live" && video.videoWidth) {
    video.pause();
    readStill(video, video.videoWidth, video.videoHeight, "cover", true);
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
  ensureModels(l).catch((e) => setStatus(e, true));
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
