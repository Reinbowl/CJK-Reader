// Camera -> OCR worker -> reading worker -> overlay of pinyin / romaji / romanization above the text.
const $ = (s) => document.querySelector(s);
const view = $("#view"), video = $("#cam"), still = $("#still"), overlay = $("#overlay");
const statusEl = $("#status"), sheet = $("#sheet"), linesEl = $("#lines");

// URL switches for performance testing: ?ep=webgpu &precision=fp32 &threads=1 &det=480
const params = new URLSearchParams(location.search);
const EP = params.get("ep") === "webgpu" ? "webgpu" : "wasm";
const PRECISION = params.get("precision") === "fp32" ? "fp32" : "int8";
const THREADS = Number(params.get("threads")) || (self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1);
const LIVE_DET = Number(params.get("det")) || 640; // detector input, longest side
const STILL_DET = 960;
const LIVE_MAX = 1280, STILL_MAX = 1920; // captured image, longest side (recogniser crops come from this)
const MODEL = { zh: "zhja", ja: "zhja", ko: "ko" };

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
    data.error ? p.reject(new Error(data.error)) : p.resolve(data);
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
const inited = ocr({ type: "init", ep: EP, precision: PRECISION, threads: THREADS });
const loaded = {};
function ensureModels(l) {
  loaded[l] ??= Promise.all([inited, ocr({ type: "load", model: MODEL[l] }), reading({ type: "load", lang: l })])
    .catch((e) => { delete loaded[l]; throw e; });
  return loaded[l];
}

// ---------- capture ----------
const grabCanvas = document.createElement("canvas");
const grabCtx = grabCanvas.getContext("2d", { willReadFrequently: true });

// Draw the part of `src` that is on screen into grabCanvas (at most maxSide px) and return its
// pixels plus the image->screen mapping. fit "cover" matches the video's CSS; "contain" shows a whole photo.
function grab(src, srcW, srcH, maxSide, fit) {
  const vw = view.clientWidth, vh = view.clientHeight;
  let sx = 0, sy = 0, sw = srcW, sh = srcH;
  if (fit === "cover") {
    const s = Math.max(vw / srcW, vh / srcH);
    sw = vw / s; sh = vh / s; sx = (srcW - sw) / 2; sy = (srcH - sh) / 2;
  }
  const k = Math.min(1, maxSide / Math.max(sw, sh));
  const w = Math.round(sw * k), h = Math.round(sh * k);
  grabCanvas.width = w; grabCanvas.height = h;
  grabCtx.drawImage(src, sx, sy, sw, sh, 0, 0, w, h);
  const s = Math.min(vw / w, vh / h);
  // A whole photo sits just under the header so the results sheet covers less of it.
  const oy = fit === "contain" ? Math.min((vh - h * s) / 2, $("header").offsetHeight) : (vh - h * s) / 2;
  return { image: grabCtx.getImageData(0, 0, w, h), map: { s, ox: (vw - w * s) / 2, oy } };
}

async function read(image, detLimit) {
  await ensureModels(lang);
  const l = lang;
  const res = await ocr({ type: "ocr", model: MODEL[l], detLimit, buffer: image.data.buffer, width: image.width, height: image.height }, [image.data.buffer]);
  const notes = res.lines.length ? (await reading({ type: "annotate", lang: l, texts: res.lines.map((x) => x.text) })).lines : [];
  return { ...res, notes, lang: l };
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
    let size = Math.min(28, thick * 0.45);
    if (!line.vertical) for (const l of labels) size = Math.min(size, (100 * pitch * l.u.n) / octx.measureText(l.text).width);
    if (size < 9) return; // too small to fit legibly: move closer, or read it in the sheet
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

function showSheet(result) {
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
  sheet.hidden = false;
}

function setStatus(msg, error = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", error);
}
const timing = (r) => `det ${r.ms.det.toFixed(0)} ms · rec ${r.ms.rec.toFixed(0)} ms · ${r.lines.length} lines · ${EP}×${THREADS} ${PRECISION}`;

// ---------- modes ----------
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
  mode = "live";
  still.hidden = true;
  sheet.hidden = true;
  setShutter("Freeze", "");
  document.body.dataset.mode = "live";
  sizeCanvas(overlay);
  if (!stream) await startCamera();
  else await video.play(); // resuming after a freeze, which only paused the video
  setStatus("Loading models…");
  liveLoop(++loopToken);
}

async function liveLoop(token) {
  while (mode === "live" && token === loopToken) {
    if (video.readyState < 2 || !video.videoWidth) { await new Promise((r) => requestAnimationFrame(r)); continue; }
    const { image, map } = grab(video, video.videoWidth, video.videoHeight, LIVE_MAX, "cover");
    try {
      const r = await read(image, LIVE_DET);
      if (mode !== "live" || token !== loopToken) break;
      draw(r, map);
      setStatus(timing(r));
    } catch (e) {
      setStatus(e.message, true);
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
}

// Read a still image (frozen camera frame or picked photo) at higher resolution and list the lines.
async function readStill(src, w, h, fit) {
  mode = "still";
  loopToken++;
  document.body.dataset.mode = "still";
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
    showSheet(r);
    setStatus(timing(r));
  } catch (e) {
    setStatus(e.message, true);
  }
}

sheet.onclick = () => sheet.classList.toggle("collapsed");

$("#shutter").onclick = () => {
  if (mode === "live" && video.videoWidth) {
    video.pause();
    readStill(video, video.videoWidth, video.videoHeight, "cover");
  } else {
    live().catch((e) => setStatus(e.message, true));
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
  ensureModels(l).catch((e) => setStatus(e.message, true));
  if (mode === "still" && lastStill) readStill(lastStill.src, lastStill.w, lastStill.h, lastStill.fit);
}
document.querySelectorAll("[data-lang]").forEach((b) => (b.onclick = () => setLang(b.dataset.lang)));

document.addEventListener("visibilitychange", () => {
  if (document.hidden) { stopCamera(); loopToken++; }
  else if (mode === "live") live().catch((e) => setStatus(e.message, true));
});
addEventListener("resize", () => { if (mode === "still" && lastStill) readStill(lastStill.src, lastStill.w, lastStill.h, lastStill.fit); });

if (!navigator.mediaDevices?.getUserMedia) $("#start-msg").textContent = "No camera access here (needs HTTPS). You can still open a photo.";
setLang(lang);
inited.catch((e) => { if (!reloading) setStatus(`Model load failed: ${e.message}`, true); });
