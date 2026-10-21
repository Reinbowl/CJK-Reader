// End-to-end checks in headless Chromium, with a fake camera playing clips made from test/images.
//   npm run e2e                 (needs docs/models and docs/vendor; screenshots go to .cache/e2e/)
// The app's self.cjkStats says what it is doing; the checks read that rather than the status bar.
import { createServer } from "node:http";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, statSync, writeFileSync, writeSync } from "node:fs";
import { extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";
import { chromium } from "playwright";

const root = fileURLToPath(new URL("../", import.meta.url));
const docs = join(root, "docs");
const out = join(root, ".cache", "e2e");
mkdirSync(out, { recursive: true });
const truth = JSON.parse(readFileSync(join(root, "test/images/truth.json"), "utf8"));

// ---------- docs/ over HTTP, like GitHub Pages: no COOP/COEP headers (sw.js adds them) ----------
const TYPES = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".webmanifest": "application/manifest+json", ".png": "image/png",
  ".wasm": "application/wasm", ".txt": "text/plain; charset=utf-8", ".gz": "application/gzip",
};
const server = createServer((req, res) => {
  let file = join(docs, decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (file.endsWith("/")) file += "index.html";
  if (!file.startsWith(docs) || !existsSync(file) || statSync(file).isDirectory()) return res.writeHead(404).end();
  res.writeHead(200, { "Content-Type": TYPES[extname(file)] ?? "application/octet-stream" });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, r));
const base = `http://localhost:${server.address().port}/`;

// ---------- fake camera clips: Y4M 4:2:0, portrait 9:16 like a phone's 1080×1920; Chromium loops them ----------
// (608×1080: the fake device crops anything taller to the 1080 px the app asks for.)
const W = 608, H = 1080, FPS = 15;

// The image scaled to fitW px wide, centred on its own corner colour and moved by (dx, dy). The
// phone viewport shows the middle ~500 px of the frame's width.
function frame(png, dx = 0, fitW = W, dy = 0) {
  const s = fitW / png.width, ih = png.height * s, top = (H - ih) / 2 + dy, left = (W - fitW) / 2 + dx;
  const rgb = new Uint8Array(W * H * 3);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const sx = Math.round((x - left) / s), sy = Math.round((y - top) / s);
      const i = sx >= 0 && sx < png.width && sy >= 0 && sy < png.height ? (sy * png.width + sx) * 4 : 0;
      rgb.set(png.data.subarray(i, i + 3), (y * W + x) * 3);
    }
  }
  return rgb;
}

function writeY4M(name, frames) {
  const path = join(out, `${name}.y4m`);
  const fd = openSync(path, "w");
  writeSync(fd, `YUV4MPEG2 W${W} H${H} F${FPS}:1 Ip A1:1 C420jpeg\n`);
  const Y = new Uint8Array(W * H), U = new Uint8Array(W * H / 4), V = new Uint8Array(W * H / 4);
  for (const rgb of frames) {
    for (let i = 0; i < W * H; i++) Y[i] = 16 + 0.257 * rgb[3 * i] + 0.504 * rgb[3 * i + 1] + 0.098 * rgb[3 * i + 2];
    for (let y = 0; y < H; y += 2) {
      for (let x = 0; x < W; x += 2) {
        const i = 3 * (y * W + x), j = (y / 2) * (W / 2) + x / 2;
        U[j] = 128 - 0.148 * rgb[i] - 0.291 * rgb[i + 1] + 0.439 * rgb[i + 2];
        V[j] = 128 + 0.439 * rgb[i] - 0.368 * rgb[i + 1] - 0.071 * rgb[i + 2];
      }
    }
    writeSync(fd, "FRAME\n");
    writeSync(fd, Y); writeSync(fd, U); writeSync(fd, V);
  }
  closeSync(fd);
  return path;
}

const image = (name) => PNG.sync.read(readFileSync(join(root, `test/images/${name}.png`)));
const repeat = (f, n) => Array(n).fill(f);
const zh = image("zh");
const pan = [...repeat(frame(zh), 20), ...Array.from({ length: 8 }, (_, k) => frame(zh, 30 * (k + 1))), ...repeat(frame(zh, 240), 30)];
let seed = 7;
const jitter = () => Math.round(8 * ((seed = (seed * 16807) % 2147483647) / 2147483647) - 4); // -4..4 px
const clips = {
  static: writeY4M("static", [frame(zh)]),
  calendar: writeY4M("calendar", [frame(image("calendar"), 0, 490)]),
  pan: writeY4M("pan", pan), // 1.3 s still, 0.5 s fast pan right, 2 s still, then a jump back
  tremor: writeY4M("tremor", Array.from({ length: 45 }, () => frame(zh, jitter(), 1100, jitter()))), // big text, shaky hand
};

// CJK fonts for the screenshots, if tools/make_test_images.py's Noto font is in .cache/fonts.
const fontDir = join(root, ".cache", "fonts");
const fontconfig = {};
if (existsSync(fontDir)) {
  fontconfig.FONTCONFIG_FILE = join(out, "fonts.conf");
  writeFileSync(fontconfig.FONTCONFIG_FILE, `<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig>
<include ignore_missing="yes">/etc/fonts/fonts.conf</include><dir>${fontDir}</dir><cachedir>${join(out, "fontcache")}</cachedir></fontconfig>\n`);
}

// ---------- helpers ----------
let failed = false;
function check(ok, what) {
  console.log(`${ok ? "ok  " : "FAIL"} ${what}`);
  if (!ok) failed = true;
}

function lev(a, b) {
  a = [...a]; b = [...b];
  let d = [...Array(b.length + 1).keys()];
  for (let i = 1; i <= a.length; i++) {
    const nd = [i];
    for (let j = 1; j <= b.length; j++) nd[j] = Math.min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (a[i - 1] !== b[j - 1]));
    d = nd;
  }
  return d[b.length];
}
// How far `got` is from being a piece of `want` (lines cut off at the screen edge are pieces), per char.
const pieceErr = (got, want) => (lev(got, want) - Math.max(0, [...want].length - [...got].length)) / Math.max(1, [...got].length);

async function open(clip, query = "", extraArgs = []) {
  const browser = await chromium.launch({
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${clip}`, ...extraArgs],
    env: { ...process.env, ...fontconfig },
  });
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true });
  await context.grantPermissions(["camera"]);
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto(base + query);
  // The first load reloads itself once the service worker can add COOP/COEP.
  for (let end = Date.now() + 60000; ;) {
    try {
      await page.waitForFunction(() => self.crossOriginIsolated && self.cjkStats, null, { timeout: end - Date.now() });
      break;
    } catch (e) {
      if (Date.now() > end || !/context|navigat/i.test(e.message)) throw e;
    }
  }
  return { browser, page, errors };
}

const stats = (page) => page.evaluate(() => structuredClone(self.cjkStats));
const until = (page, fn, arg, timeout = 60000) => page.waitForFunction(fn, arg, { timeout, polling: 50 });
const shot = (page, name) => page.screenshot({ path: join(out, `${name}.png`) });

async function startLive(page) {
  await page.click("#go");
  await until(page, () => self.cjkStats.reads >= 1);
  return Date.now();
}

// ---------- scenarios ----------
async function staticScene() {
  console.log("-- static sign: one read, then idle");
  const { browser, page, errors } = await open(clips.static);
  const t0 = await startLive(page);
  const first = await stats(page);
  const lines = first.lines;
  const cam = await page.evaluate(() => `${document.querySelector("#cam").videoWidth}×${document.querySelector("#cam").videoHeight}`);
  console.log(`    first read: ${lines} lines, ${first.skipped} small, det ${first.ms.det.toFixed(0)} ms rec ${first.ms.rec.toFixed(0)} ms, ${first.backend.backend}, camera ${cam}`);
  check(lines >= 4, `reads at least 4 lines on the sign (${lines})`);
  check(first.backend.backend === "wasm" && !first.backend.fallback, "WASM backend reported");
  await page.waitForTimeout(3000);
  const later = await stats(page);
  check(later.reads === first.reads, `no further reads while nothing changes (${later.reads - first.reads} extra)`);
  check(later.gate === "idle", `gate idle (${later.gate})`);
  check(later.busy / (Date.now() - t0) < 0.5, `busy ${(100 * later.busy / (Date.now() - t0)).toFixed(0)} % of the time`);
  const faded = await page.evaluate(() => document.querySelector("#overlay").classList.contains("faded"));
  check(!faded, "overlay shown");
  await shot(page, "static-live");
  await page.click('[data-lang="ja"]');
  await until(page, (n) => self.cjkStats.reads > n && self.cjkStats.gate === "idle", later.reads);
  check(true, "switching language re-reads the unchanged view");
  await page.click('[data-lang="zh"]');
  await until(page, (n) => self.cjkStats.reads > n + 1 && self.cjkStats.gate === "idle", later.reads);
  const back = await stats(page);

  // Shutter: still mode reads the frozen frame and lists the lines.
  await page.click("#shutter");
  await until(page, (n) => self.cjkStats.mode === "still" && self.cjkStats.reads > n, back.reads);
  // The sheet's lines without their ruby text.
  const listed = await page.evaluate(() => [...document.querySelectorAll("#lines li .ruby")]
    .map((e) => [...e.childNodes].map((n) => (n.nodeName === "RUBY" ? n.firstChild.textContent : n.textContent)).join("")));
  const want = truth.zh.lines.map((l) => l.replace(/\s/g, ""));
  // Lines cut off at the screen edge can end in a misread half character, so not all need to match.
  const good = listed.filter((g) => Math.min(...want.map((w) => pieceErr(g.replace(/\s/g, ""), w))) <= 0.1);
  console.log(`    still read: ${listed.join(" | ")}`);
  check(good.length >= 4, `still read: ${good.length} of ${listed.length} lines are (pieces of) the sign's lines`);
  const lab = (await stats(page)).labels;
  console.log(`    labels: ${JSON.stringify(lab)}`);
  check(lab.lines === listed.length && lab.overlaps === 0, `every line has its readings drawn (${lab.lines}/${listed.length}), none overlapping`);
  const sheetHidden = () => page.evaluate(() => document.querySelector("#sheet").hidden);
  check(await sheetHidden(), "the list stays closed after the shutter");
  await shot(page, "static-still");
  await page.tap("#list");
  check(!(await sheetHidden()), "list button opens the list");
  await shot(page, "static-still-list");
  await page.tap("#sheet");
  check(await sheetHidden(), "tapping the list closes it");
  await page.click("#shutter");
  await until(page, () => self.cjkStats.mode === "live");
  check(true, "shutter returns to live");

  // Photo from the library, in Japanese.
  await page.click('[data-lang="ja"]');
  const before = (await stats(page)).reads;
  await page.setInputFiles("#file", join(root, "test/images/ja.png"));
  await until(page, (n) => self.cjkStats.mode === "still" && self.cjkStats.reads > n, before);
  await page.tap("#list");
  const ja = await page.evaluate(() => document.querySelector("#lines").innerText);
  check(ja.includes("tōkyō eki wa kochira desu"), "photo read in Japanese with romaji");
  await shot(page, "photo-ja");
  check(!errors.length, `no page errors${errors.length ? `: ${errors.join(" | ")}` : ""}`);
  await browser.close();
}

async function calendarScene() {
  console.log("-- calendar: many small boxes");
  const { browser, page, errors } = await open(clips.calendar);
  // Dev panel: three taps on the status line, then WebGPU (headless Chromium has no adapter, so the
  // app must say it fell back), then back to WASM. Both reload the app; the panel stays open.
  for (let k = 0; k < 3; k++) await page.tap("#status");
  check(await page.isVisible("#devbar"), "three taps on the status line open the dev panel");
  for (const [label, want] of [["WebGPU", "webgpu"], ["WASM", "wasm"]]) {
    await Promise.all([page.waitForNavigation(), page.click(`#devbar button:text-is("${label}")`)]);
    await until(page, () => self.cjkStats?.backend?.backend);
    const b = (await stats(page)).backend;
    console.log(`    dev panel ${label}: ${JSON.stringify(b)}`);
    check(want === "wasm" ? b.backend === "wasm" && !b.fallback : b.backend === "webgpu" || !!b.fallback, `${label} chosen, and any fallback reported`);
  }
  await startLive(page);
  const s = await stats(page);
  const debug = await page.evaluate(() => document.querySelector("#debug").textContent);
  console.log(`    debug: ${debug.split("\n").join(" / ")}`);
  check(/gate \w+ still \d+/.test(debug) && /safe [\d./]+/.test(debug), "debug readout");
  console.log(`    first read: ${s.lines} lines, ${s.skipped} small, ${s.recognised} recognised, det ${s.ms.det.toFixed(0)} ms rec ${s.ms.rec.toFixed(0)} ms`);
  check(s.skipped > 0, "small lines (lunar days) are skipped in live mode");
  check(s.recognised + s.skipped >= 50, `boxes found (${s.recognised + s.skipped})`);
  await shot(page, "calendar-live");
  await page.click("#shutter");
  await until(page, (n) => self.cjkStats.mode === "still" && self.cjkStats.reads > n, s.reads);
  const t = await stats(page);
  console.log(`    still read: ${t.lines} lines, ${t.skipped} small, det ${t.ms.det.toFixed(0)} ms rec ${t.ms.rec.toFixed(0)} ms`);
  check(t.skipped === 0 && t.lines >= s.lines, "still mode skips nothing");
  const withReading = await page.evaluate(() => document.querySelectorAll("#lines li:has(rt)").length);
  console.log(`    labels: ${JSON.stringify(t.labels)}`);
  check(t.labels.lines === withReading && t.labels.overlaps === 0, `every line with a reading is labelled (${t.labels.lines}/${withReading}), none overlapping`);
  await shot(page, "calendar-still");
  check(!errors.length, `no page errors${errors.length ? `: ${errors.join(" | ")}` : ""}`);
  await browser.close();
}

// What the gate and overlay do over `ms`, sampled every 50 ms.
const watch = (page, ms) => page.evaluate(async (ms) => {
  const out = [];
  for (const t0 = performance.now(); performance.now() - t0 < ms; await new Promise((r) => setTimeout(r, 50))) {
    const s = self.cjkStats;
    out.push({ gate: s.gate, fast: s.fast, reads: s.reads, faded: document.querySelector("#overlay").classList.contains("faded") });
  }
  return out;
}, ms);

async function panScene() {
  console.log("-- fast pan: no reads while moving, overlay fades, one read once settled");
  const { browser, page, errors } = await open(clips.pan);
  await startLive(page);
  // Two loops of the clip (3.9 s each).
  const log = await watch(page, 9000);
  const reads = log.at(-1).reads - log[0].reads;
  const fast = log.filter((e) => e.fast);
  console.log(`    ${reads} reads in 9 s; fast ${fast.length} samples, faded while fast ${fast.filter((e) => e.faded).length}`);
  check(fast.length >= 5, "gate sees the pan as a fast move");
  check(fast.filter((e) => e.faded).length >= fast.length * 0.8, "overlay fades during the fast move");
  check(reads >= 2 && reads <= 8, `reads only after settling (${reads} in 9 s; continuous reading would be ~30)`);
  // Snap while a live read is still running: the live read is cancelled (or finishes first), the
  // still read runs after it, never alongside it.
  await until(page, () => self.cjkStats.mode === "live" && self.cjkStats.reading > 0);
  const before = await stats(page);
  await page.click("#shutter");
  await until(page, (n) => self.cjkStats.mode === "still" && self.cjkStats.reads > n && !self.cjkStats.reading, before.reads);
  const after = await stats(page);
  console.log(`    snap during a live read: ${after.cancelled - before.cancelled} live read cancelled, still read ${after.lines} lines`);
  check(after.lines > 0, "snap during a live read gives a still read");
  check(!errors.length, `no page errors${errors.length ? `: ${errors.join(" | ")}` : ""}`);
  await browser.close();
}

async function tremorScene() {
  console.log("-- big text, shaky hand: labels stay up and follow, few re-reads");
  const { browser, page, errors } = await open(clips.tremor);
  await startLive(page);
  const log = await watch(page, 6000);
  const reads = log.at(-1).reads - log[0].reads;
  const shown = log.filter((e) => !e.faded).length / log.length;
  const s = await stats(page);
  console.log(`    ${reads} re-reads in 6 s, labels shown ${(100 * shown).toFixed(0)} % of the time, ${s.lines} lines`);
  await shot(page, "tremor-live");
  check(shown >= 0.8, "labels stay up through the tremor");
  check(reads <= 2, "tremor doesn't trigger re-reads");
  check(!errors.length, `no page errors${errors.length ? `: ${errors.join(" | ")}` : ""}`);
  await browser.close();
}

// WebGPU through SwiftShader (software, so a read takes ~10 s): only with E2E_WEBGPU=1. Snapping
// during a live read used to run two reads at once on the GPU, which fails on iOS ("Failed to
// download data from buffer") and hangs SwiftShader.
async function webgpuScene() {
  console.log("-- WebGPU (SwiftShader): snap during a live read");
  const { browser, page, errors } = await open(clips.pan, "?ep=webgpu", ["--enable-unsafe-webgpu", "--enable-unsafe-swiftshader"]);
  page.setDefaultTimeout(240000);
  await page.click("#go");
  await until(page, () => self.cjkStats.reads >= 1, null, 240000);
  const first = await stats(page);
  console.log(`    first read: ${first.lines} lines, det ${first.ms.det.toFixed(0)} ms rec ${first.ms.rec.toFixed(0)} ms, ${JSON.stringify(first.backend)}`);
  check(first.backend.backend === "webgpu" && !first.backend.fallback, "runs on WebGPU");
  await until(page, () => self.cjkStats.mode === "live" && self.cjkStats.reading > 0, null, 240000);
  await page.click("#shutter");
  await until(page, (n) => self.cjkStats.mode === "still" && self.cjkStats.reads > n && !self.cjkStats.reading, first.reads, 240000);
  const after = await stats(page);
  console.log(`    snap: ${after.cancelled} live read cancelled, still read ${after.lines} lines, ${JSON.stringify(after.backend)}`);
  check(after.lines > 0 && after.backend.backend === "webgpu" && !after.backend.fallback, "snap during a live read works on WebGPU, no fallback");
  check(!errors.length, `no page errors${errors.length ? `: ${errors.join(" | ")}` : ""}`);
  await browser.close();
}

async function cartonScene() {
  console.log("-- stacked headlines (photo): big labels, over the text when there's no gap");
  const { browser, page, errors } = await open(clips.static);
  await page.click('[data-lang="ja"]');
  await page.setInputFiles("#file", join(root, "test/images/carton.png"));
  await until(page, () => self.cjkStats.mode === "still" && self.cjkStats.reads >= 1, null, 120000);
  const s = await stats(page);
  console.log(`    ${s.lines} lines, labels: ${JSON.stringify(s.labels)}`);
  await shot(page, "carton-still");
  check(s.labels.lines === truth.carton.lines.length && s.labels.overlaps === 0, `every line labelled (${s.labels.lines}/${truth.carton.lines.length}), none overlapping`);
  check(s.labels.bigShrink >= 0.75, `big lines keep at least 3/4 of their label size (${s.labels.bigShrink.toFixed(2)})`);
  check(s.labels.over >= 1, "a big line with no room above or below gets its labels over its text");
  check(!errors.length, `no page errors${errors.length ? `: ${errors.join(" | ")}` : ""}`);
  await browser.close();
}

async function zoomScene() {
  console.log("-- zoom: slider and pinch live, pinch and pan the frozen frame, double-tap resets");
  const { browser, page, errors } = await open(clips.calendar);
  const cdp = await page.context().newCDPSession(page);
  const touch = (type, pts) => cdp.send("Input.dispatchTouchEvent", { type, touchPoints: pts.map(([x, y], id) => ({ x, y, id })) });
  const pinch = async (cx, cy, d0, d1) => {
    await touch("touchStart", [[cx - d0 / 2, cy], [cx + d0 / 2, cy]]);
    for (let i = 1; i <= 10; i++) {
      const d = d0 + ((d1 - d0) * i) / 10;
      await touch("touchMove", [[cx - d / 2, cy], [cx + d / 2, cy]]);
    }
    await touch("touchEnd", []);
  };
  const doubleTap = async () => { await page.touchscreen.tap(195, 420); await page.touchscreen.tap(195, 420); };
  await startLive(page);
  const one = await stats(page);

  await page.$eval("#zoom", (el) => { el.value = "2"; el.dispatchEvent(new Event("input", { bubbles: true })); });
  await until(page, (n) => self.cjkStats.reads > n && self.cjkStats.gate === "idle", one.reads);
  const two = await stats(page);
  console.log(`    1×: ${one.lines} lines, labels ${one.labels.min.toFixed(1)}–${one.labels.max.toFixed(1)} px · 2×: ${two.lines} lines, labels ${two.labels.min.toFixed(1)}–${two.labels.max.toFixed(1)} px · ${JSON.stringify(two.zoom)}`);
  check(two.zoom.live === 2 && two.zoom.digital === 2, "slider sets 2× (digital: this camera has no zoom of its own)");
  // (Its "small" lines are slivers of characters cut off at the screen edges.)
  check(two.labels.max > one.labels.max * 1.3, "the zoomed read's labels are bigger");
  await shot(page, "zoom-live-2x");
  await pinch(195, 420, 100, 160);
  const p = await stats(page);
  check(p.zoom.live > 2.5, `pinching out zooms in (${p.zoom.live}×)`);
  await doubleTap();
  await until(page, () => self.cjkStats.zoom.live === 1);
  check(true, "double-tap resets the live zoom");

  const before = await stats(page);
  await page.click("#shutter");
  await until(page, (n) => self.cjkStats.mode === "still" && self.cjkStats.reads > n && !self.cjkStats.reading, before.reads);
  const st = await stats(page);
  await pinch(195, 420, 80, 240);
  await until(page, () => self.cjkStats.zoom.still > 2);
  const zs = await stats(page);
  console.log(`    still 1×: labels ${st.labels.min.toFixed(1)}–${st.labels.max.toFixed(1)} px · ${zs.zoom.still.toFixed(2)}×: ${zs.labels.min.toFixed(1)}–${zs.labels.max.toFixed(1)} px`);
  console.log(`    zoomed labels: ${JSON.stringify(zs.labels)}`);
  const same = ["lines", "labels", "crowded", "over", "collisions"].every((k) => zs.labels[k] === st.labels[k]);
  check(same && Math.abs(zs.labels.max - st.labels.max * zs.zoom.still) < 0.5, "zooming the frozen frame scales its boxes and labels like a page, same layout");
  check(zs.reads === st.reads, "without reading it again");
  await shot(page, "zoom-still");
  await touch("touchStart", [[150, 400]]);
  for (let i = 1; i <= 5; i++) await touch("touchMove", [[150 + 20 * i, 400 + 20 * i]]);
  await touch("touchEnd", []);
  await until(page, (x) => self.cjkStats.zoom.pan[0] !== x, zs.zoom.pan[0]);
  check(true, "dragging pans the zoomed frame");
  await doubleTap();
  await until(page, () => self.cjkStats.zoom.still === 1);
  check(true, "double-tap resets the still zoom");
  check(!errors.length, `no page errors${errors.length ? `: ${errors.join(" | ")}` : ""}`);
  await browser.close();
}

// E2E_ONLY=static,pan runs just those; webgpu only runs when asked for (E2E_WEBGPU=1 or E2E_ONLY).
const scenes = { webgpu: webgpuScene, static: staticScene, calendar: calendarScene, carton: cartonScene, zoom: zoomScene, pan: panScene, tremor: tremorScene };
const only = process.env.E2E_ONLY?.split(",");
try {
  for (const [name, scene] of Object.entries(scenes)) {
    if (only ? only.includes(name) : name !== "webgpu" || process.env.E2E_WEBGPU === "1") await scene();
  }
} catch (e) {
  console.log("FAIL", e);
  failed = true;
} finally {
  server.close();
}
process.exit(failed ? 1 : 0);
