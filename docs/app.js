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
// Labels are drawn at LABEL_SCALE × the line's thickness and not at all below MIN_LABEL px, so live
// mode doesn't recognise lines thinner than MIN_LINE px on screen.
const LABEL_SCALE = 0.45, MIN_LABEL = 9, MIN_LINE = MIN_LABEL / LABEL_SCALE;
const MIN_GAP = 250; // live mode: ms between reads, at least, and at least as long as the last read took

// What the app is doing, for the e2e tests (test/e2e.test.mjs) and ?debug.
const stats = (self.cjkStats = { mode: "idle", backend: "", reads: 0, lines: 0, skipped: 0, recognised: 0, busy: 0, ms: null, gate: null });

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
    data.error ? p.reject(Object.assign(new Error(data.error), { where: data.where })) : p.resolve(data);
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

const ocr = rpc(new Worker("ocr-worker.js", { type: "module" }), onProgress);
const reading = rpc(new Worker("reading-worker.js"), onProgress);
const inited = ocr({ type: "init", ep: EP, precision: PRECISION, threads: THREADS, spin: SPIN });
let backend = EP;
inited.then((r) => {
  backend = r.fallback ? `${r.backend} (no webgpu)` : r.backend;
  stats.backend = { ...r };
  if (r.fallback) console.warn(r.fallback);
}, () => {});
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
  const res = await ocr({ type: "ocr", model: MODEL[l], detLimit, minLineH, buffer: image.data.buffer, width: image.width, height: image.height }, [image.data.buffer]);
  const notes = res.lines.length ? (await reading({ type: "annotate", lang: l, texts: res.lines.map((x) => x.text) })).lines : [];
  Object.assign(stats, { reads: stats.reads + 1, lines: res.lines.length, skipped: res.skipped.length, recognised: stats.recognised + res.recognised, ms: res.ms });
  return { ...res, notes, lang: l };
}

// ---------- motion gate (live mode) ----------
// Every GATE_MS the on-screen part of the video is shrunk to a grey thumbnail THUMB_W px wide,
// normalised (mean 0, sd 1) so exposure changes don't count, and compared with the previous one
// (motion) and with the one taken when the overlay's frame was read (drift). Reading starts once the
// view has been still for SETTLE samples and has changed since the last read; while the camera
// moves the overlay fades out, and small drift moves it along with the text.
const GATE_MS = 100, THUMB_W = 36, SETTLE = 3, SEARCH = 3;
const MOVE_PX = 0.5, MOVE_ERR = 0.25; // per sample: thumbnail px, mean abs difference in sd units
const CHANGE_PX = 1, CHANGE_ERR = 0.2; // since the last read
const thumbCanvas = document.createElement("canvas");
const thumbCtx = thumbCanvas.getContext("2d", { willReadFrequently: true });
const gate = { state: "moving", still: 0, prev: null, read: null, motion: null, drift: null, timer: 0 };

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
  const moving = !m || Math.hypot(m.dx, m.dy) > MOVE_PX || m.err > MOVE_ERR;
  gate.still = moving ? 0 : gate.still + 1;
  const d = (gate.drift = compare(gate.read, cur));
  const same = d && d.err < CHANGE_ERR; // still the scene the overlay was read from, maybe shifted
  gate.state = gate.still < SETTLE ? "moving" : !same || Math.hypot(d.dx, d.dy) > CHANGE_PX ? "ready" : "idle";
  const show = !moving && same;
  overlay.classList.toggle("faded", !show);
  if (show) {
    const k = view.clientWidth / cur.w;
    overlay.style.transform = `translate(${d.dx * k}px, ${d.dy * k}px)`;
  }
  stats.gate = gate.state;
  if (DEBUG) debugReadout();
}

function startGate() {
  clearInterval(gate.timer);
  Object.assign(gate, { state: "moving", still: 0, prev: null, read: null, motion: null, drift: null });
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

function draw(result, map) {
  sizeCanvas(overlay);
  const P = (x, y) => [map.ox + x * map.s, map.oy + y * map.s];
  // Lines too small to label (live mode doesn't read them): a faint dashed hint to move closer.
  octx.setLineDash([4, 4]);
  octx.lineWidth = 1;
  octx.strokeStyle = "rgba(255,255,255,0.4)";
  for (const box of result.skipped) {
    octx.beginPath();
    box.forEach(([x, y], i) => (i ? octx.lineTo : octx.moveTo).call(octx, ...P(x, y)));
    octx.closePath();
    octx.stroke();
  }
  octx.setLineDash([]);
  result.lines.forEach((line, k) => {
    const note = result.notes[k];
    octx.beginPath();
    line.box.forEach(([x, y], i) => (i ? octx.lineTo : octx.moveTo).call(octx, ...P(x, y)));
    octx.closePath();
    octx.lineWidth = 1.5;
    octx.strokeStyle = "rgba(255,255,255,0.75)";
    octx.stroke();
    if (!note) return;

    const bl = Math.hypot(line.B[0], line.B[1]) || 1;
    const bn = [line.B[0] / bl, line.B[1] / bl]; // unit vector "down" across the line
    const thick = bl * map.s;
    const len = Math.hypot(line.A[0], line.A[1]) * map.s;
    const angle = line.vertical ? 0 : Math.atan2(line.A[1], line.A[0]);
    const labels = note.units
      .map((u) => ({ u, text: labelFor(u, result.lang), cs: line.chars.slice(u.i, u.i + u.n) }))
      .filter((l) => l.text && l.cs.length);
    if (!labels.length) return;

    // Largest font (up to 28px) at which each label fits over its own characters.
    const font = (px) => `600 ${px}px system-ui, -apple-system, sans-serif`;
    octx.font = font(100);
    // Character pitch: median gap between neighbouring labelled characters (digits/latin are narrower).
    const labelled = new Array(line.chars.length).fill(false);
    for (const { u } of labels) labelled.fill(true, u.i, Math.min(u.i + u.n, line.chars.length));
    const gaps = [];
    for (let k = 0; k + 1 < line.chars.length; k++) {
      const a = line.chars[k], b = line.chars[k + 1];
      if (labelled[k] && labelled[k + 1]) gaps.push(Math.hypot(b.x - a.x, b.y - a.y) * map.s);
    }
    gaps.sort((a, b) => a - b);
    const pitch = gaps.length ? gaps[gaps.length >> 1] : len / Math.max(1, line.chars.length);
    let size = Math.min(28, thick * LABEL_SCALE);
    if (!line.vertical) for (const l of labels) size = Math.min(size, (100 * pitch * l.u.n) / octx.measureText(l.text).width);
    if (size < MIN_LABEL) return; // too small to fit legibly: move closer, or read it in the sheet
    octx.font = font(size);
    octx.textBaseline = "middle";
    // Horizontal text: labels on a band above the line. Vertical text: to the right, like furigana.
    const off = line.vertical ? thick / 2 + 4 : thick / 2 + size * 0.75;
    if (!line.vertical) {
      const [cx, cy] = P((line.box[0][0] + line.box[2][0]) / 2, (line.box[0][1] + line.box[2][1]) / 2);
      octx.save();
      octx.translate(cx - bn[0] * off, cy - bn[1] * off);
      octx.rotate(angle);
      octx.fillStyle = "rgba(0,0,0,0.6)";
      octx.fillRect(-len / 2, -size * 0.7, len, size * 1.4);
      octx.restore();
    }
    for (const { text, cs } of labels) {
      let [x, y] = P(cs.reduce((a, c) => a + c.x, 0) / cs.length, cs.reduce((a, c) => a + c.y, 0) / cs.length);
      x -= bn[0] * off; y -= bn[1] * off;
      octx.save();
      octx.translate(x, y);
      octx.rotate(angle);
      if (line.vertical) {
        const w = octx.measureText(text).width;
        octx.fillStyle = "rgba(0,0,0,0.6)";
        octx.fillRect(-3, -size * 0.65, w + 6, size * 1.3);
      }
      octx.textAlign = line.vertical ? "left" : "center";
      octx.fillStyle = "#ffe066";
      octx.fillText(text, 0, 0);
      octx.restore();
    }
  });
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
    `gate ${gate.state} still ${gate.still} motion ${f(gate.motion)} drift ${f(gate.drift)}`,
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
      setStatus(e, true);
      await sleep(1000);
    }
  }
}

// Read a still image (frozen camera frame or picked photo) at higher resolution and list the lines.
async function readStill(src, w, h, fit) {
  setMode("still");
  loopToken++;
  stopGate();
  setShutter("Back to live camera", "Live");
  const { image, map } = grab(src, w, h, STILL_MAX, fit);
  const sctx = sizeCanvas(still);
  sctx.fillStyle = "#000";
  sctx.fillRect(0, 0, view.clientWidth, view.clientHeight);
  sctx.drawImage(grabCanvas, map.ox, map.oy, image.width * map.s, image.height * map.s);
  still.hidden = false;
  lastStill = { src, w, h, fit };
  sizeCanvas(overlay);
  setStatus("Reading…");
  try {
    const r = await read(image, STILL_DET);
    if (mode !== "still") return;
    draw(r, map);
    fillSheet(r);
    setStatus(timing(r));
  } catch (e) {
    setStatus(e, true);
  }
}

sheet.onclick = () => toggleSheet(false);
$("#list").onclick = () => toggleSheet();

$("#shutter").onclick = () => {
  if (mode === "live" && video.videoWidth) {
    video.pause();
    readStill(video, video.videoWidth, video.videoHeight, "cover");
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
inited.then(debugReadout, () => {});
setLang(lang);
inited.catch((e) => { if (!reloading) setStatus(`Model load failed: ${e.message}`, true); });
