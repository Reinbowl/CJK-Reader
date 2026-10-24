// PP-OCR pipeline (PP-OCRv6 detector and recognisers, PP-OCRv5 for Korean): DB text detection ->
// crop -> CTC recognition.
// Pure JS on RGBA pixel buffers so it runs the same in the web worker and in Node tests.
// Mirrors PaddleOCR's pre/post-processing (see tools/ocr_ref.py for the reference version).

const DET_MEAN = [0.485, 0.456, 0.406];
const DET_STD = [0.229, 0.224, 0.225];
const REC_H = 48;
// PaddleOCR pads every crop to at least 320 px wide; short boxes (calendar dates, one-word signs)
// read the same without it and cost a fraction. Batches hold as many crops as fit REC_COLS columns.
const REC_MIN_W = 64;
const REC_COLS = 1280;

// Recogniser input width for a crop (or batch) of the given width/height ratio.
const recWidth = (ratio) => Math.max(REC_MIN_W, Math.ceil((REC_H * ratio) / 32) * 32);

// Bilinear sample of an RGBA image, edge-clamped. Writes B,G,R (the models were trained on BGR).
function sampleBGR(img, x, y, out) {
  const { data, width: w, height: h } = img;
  x = Math.min(Math.max(x, 0), w - 1);
  y = Math.min(Math.max(y, 0), h - 1);
  const x0 = x | 0, y0 = y | 0;
  const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
  const fx = x - x0, fy = y - y0;
  const i00 = (y0 * w + x0) * 4, i01 = (y0 * w + x1) * 4, i10 = (y1 * w + x0) * 4, i11 = (y1 * w + x1) * 4;
  for (let c = 0; c < 3; c++) {
    const top = data[i00 + c] + (data[i01 + c] - data[i00 + c]) * fx;
    const bot = data[i10 + c] + (data[i11 + c] - data[i10 + c]) * fx;
    out[2 - c] += top + (bot - top) * fy;
  }
}

// Fill a CHW float tensor region by sampling src(x, y) = O + fx*A + fy*B over an outW x outH grid,
// with k x k supersampling so large downscales don't alias.
function warpInto(img, O, A, B, outW, outH, dst, dstW, dstH, norm, k) {
  const plane = dstW * dstH;
  const px = [0, 0, 0];
  const inv = 1 / (k * k);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      px[0] = px[1] = px[2] = 0;
      for (let sy = 0; sy < k; sy++) {
        const fy = (y + (sy + 0.5) / k) / outH;
        for (let sx = 0; sx < k; sx++) {
          const fx = (x + (sx + 0.5) / k) / outW;
          sampleBGR(img, O[0] + fx * A[0] + fy * B[0] - 0.5, O[1] + fx * A[1] + fy * B[1] - 0.5, px);
        }
      }
      const o = y * dstW + x;
      for (let c = 0; c < 3; c++) dst[c * plane + o] = norm(px[c] * inv, c);
    }
  }
}

// ---------- detection ----------

export function detInput(img, limit = 960) {
  const r = Math.min(1, limit / Math.max(img.width, img.height));
  const w = Math.max(32, Math.round((img.width * r) / 32) * 32);
  const h = Math.max(32, Math.round((img.height * r) / 32) * 32);
  const data = new Float32Array(3 * w * h);
  const k = Math.min(3, Math.max(1, Math.round(img.width / w)));
  warpInto(img, [0, 0], [img.width, 0], [0, img.height], w, h, data, w, h,
    (v, c) => (v / 255 - DET_MEAN[c]) / DET_STD[c], k);
  return { data, w, h, sx: img.width / w, sy: img.height / h };
}

function convexHull(pts) {
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

// Minimum-area rectangle of a convex hull (rotating calipers over hull edges).
function minAreaRect(hull) {
  let best = null;
  const n = hull.length;
  for (let i = 0; i < n; i++) {
    const a = hull[i], b = hull[(i + 1) % n];
    let ux = b[0] - a[0], uy = b[1] - a[1];
    const len = Math.hypot(ux, uy);
    if (len === 0) { ux = 1; uy = 0; } else { ux /= len; uy /= len; }
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (const p of hull) {
      const u = p[0] * ux + p[1] * uy, v = -p[0] * uy + p[1] * ux;
      if (u < u0) u0 = u; if (u > u1) u1 = u;
      if (v < v0) v0 = v; if (v > v1) v1 = v;
    }
    const area = (u1 - u0) * (v1 - v0);
    if (!best || area < best.area) best = { area, ux, uy, u0, u1, v0, v1 };
  }
  const { ux, uy, u0, u1, v0, v1 } = best;
  const cu = (u0 + u1) / 2, cv = (v0 + v1) / 2;
  return { cx: cu * ux - cv * uy, cy: cu * uy + cv * ux, w: u1 - u0, h: v1 - v0, ux, uy };
}

function rectCorners({ cx, cy, w, h, ux, uy }) {
  const ax = (ux * w) / 2, ay = (uy * w) / 2, bx = (-uy * h) / 2, by = (ux * h) / 2;
  const p = [[cx - ax - bx, cy - ay - by], [cx + ax - bx, cy + ay - by], [cx + ax + bx, cy + ay + by], [cx - ax + bx, cy - ay + by]];
  // PaddleOCR's ordering: top-left, top-right, bottom-right, bottom-left.
  p.sort((a, b) => a[0] - b[0]);
  const [l0, l1] = p[0][1] <= p[1][1] ? [p[0], p[1]] : [p[1], p[0]];
  const [r0, r1] = p[2][1] <= p[3][1] ? [p[2], p[3]] : [p[3], p[2]];
  return [l0, r0, r1, l1];
}

// Mean probability inside a rotated rectangle.
function rectScore(prob, w, h, r) {
  const pts = rectCorners(r);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.max(0, Math.floor(Math.min(...xs))), x1 = Math.min(w - 1, Math.ceil(Math.max(...xs)));
  const y0 = Math.max(0, Math.floor(Math.min(...ys))), y1 = Math.min(h - 1, Math.ceil(Math.max(...ys)));
  let sum = 0, n = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - r.cx, dy = y - r.cy;
      if (Math.abs(dx * r.ux + dy * r.uy) <= r.w / 2 + 0.5 && Math.abs(-dx * r.uy + dy * r.ux) <= r.h / 2 + 0.5) {
        sum += prob[y * w + x];
        n++;
      }
    }
  }
  return n ? sum / n : 0;
}

// Defaults: DBPostProcess settings of the PP-OCRv6 tiny detector (its inference.yml).
export function detBoxes(prob, w, h, sx, sy, { thresh = 0.2, boxThresh = 0.4, unclipRatio = 1.4, maxBoxes = 300 } = {}) {
  const label = new Int32Array(w * h);
  const stack = new Int32Array(w * h);
  const boxes = [];
  let next = 0;
  for (let start = 0; start < w * h && boxes.length < maxBoxes; start++) {
    if (label[start] || prob[start] <= thresh) continue;
    // 8-connected flood fill; keep each row's leftmost/rightmost pixel for the hull.
    next++;
    let sp = 0;
    stack[sp++] = start;
    label[start] = next;
    const rowMin = new Map(), rowMax = new Map();
    while (sp) {
      const i = stack[--sp], x = i % w, y = (i / w) | 0;
      if (!(rowMin.get(y) <= x)) rowMin.set(y, x);
      if (!(rowMax.get(y) >= x)) rowMax.set(y, x);
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const j = yy * w + xx;
          if (!label[j] && prob[j] > thresh) { label[j] = next; stack[sp++] = j; }
        }
      }
    }
    const pts = [];
    for (const [y, x] of rowMin) { pts.push([x, y]); if (rowMax.get(y) !== x) pts.push([rowMax.get(y), y]); }
    const hull = convexHull(pts);
    if (hull.length < 3) continue; // single pixel or a 1-px line: camera noise, no area
    const r = minAreaRect(hull);
    if (Math.min(r.w, r.h) < 3) continue;
    if (rectScore(prob, w, h, r) < boxThresh) continue;
    // "Unclip": grow the shrunk text kernel back out, as PaddleOCR does with a polygon offset.
    const d = (r.w * r.h * unclipRatio) / (2 * (r.w + r.h));
    r.w += 2 * d;
    r.h += 2 * d;
    if (Math.min(r.w, r.h) < 5) continue;
    boxes.push(rectCorners(r).map(([x, y]) => [x * sx, y * sy]));
  }
  return sortBoxes(boxes);
}

// Top-to-bottom, left-to-right, treating boxes within 10px vertically as the same row.
function sortBoxes(boxes) {
  boxes.sort((a, b) => a[0][1] - b[0][1] || a[0][0] - b[0][0]);
  for (let i = 0; i < boxes.length - 1; i++) {
    for (let j = i; j >= 0; j--) {
      if (Math.abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10 && boxes[j + 1][0][0] < boxes[j][0][0]) {
        [boxes[j], boxes[j + 1]] = [boxes[j + 1], boxes[j]];
      } else break;
    }
  }
  return boxes;
}

// ---------- recognition ----------

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];

// Crop frame for a box: origin O and axes A (reading direction) and B (down the glyphs).
// Tall boxes are read top-to-bottom, like PaddleOCR's rot90 of vertical crops.
export function cropGeometry(pts) {
  const [p0, p1, p2, p3] = pts;
  const cw = Math.max(dist(p0, p1), dist(p2, p3));
  const ch = Math.max(dist(p0, p3), dist(p1, p2));
  if (ch / cw >= 1.5) return { O: p1, A: sub(p2, p1), B: sub(p0, p1), w: ch, h: cw, vertical: true };
  return { O: p0, A: sub(p1, p0), B: sub(p3, p0), w: cw, h: ch, vertical: false };
}

export function recInput(img, geoms) {
  const W = recWidth(Math.max(...geoms.map((g) => g.w / g.h)));
  const data = new Float32Array(geoms.length * 3 * REC_H * W); // zero = mid-grey padding after normalisation
  const widths = geoms.map((g, i) => {
    const tw = Math.max(1, Math.min(W, Math.ceil((REC_H * g.w) / g.h)));
    const k = Math.min(4, Math.max(1, Math.round(g.h / REC_H)));
    const dst = data.subarray(i * 3 * REC_H * W, (i + 1) * 3 * REC_H * W);
    warpInto(img, g.O, g.A, g.B, tw, REC_H, dst, W, REC_H, (v) => v / 127.5 - 1, k);
    return tw;
  });
  return { data, W, widths };
}

// ---------- fingerprints (live mode: has the text in a box changed?) ----------

const FP_H = 10, FP_MAX = 160;

// A box's text resampled to FP_H px high (as wide as that makes it, at most FP_MAX), as an edge map:
// shapes rather than colours, so a karaoke colour sweep over the same words isn't a change.
// Normalised (mean 0, sd 1); len is the box's length in px.
export function fingerprint(img, pts) {
  const g = cropGeometry(pts);
  const h = FP_H, w = Math.max(4, Math.min(FP_MAX, Math.round((FP_H * g.w) / g.h)));
  const rgb = new Float32Array(3 * w * h);
  warpInto(img, g.O, g.A, g.B, w, h, rgb, w, h, (v) => v, Math.min(4, Math.max(1, Math.round(g.h / h))));
  const grey = (x, y) => rgb[y * w + x] + rgb[w * h + y * w + x] + rgb[2 * w * h + y * w + x];
  const e = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = grey(x, y);
      e[y * w + x] = Math.abs(grey(Math.min(w - 1, x + 1), y) - v) + Math.abs(grey(x, Math.min(h - 1, y + 1)) - v);
    }
  }
  const mean = e.reduce((a, v) => a + v, 0) / e.length;
  const sd = Math.max(3, Math.sqrt(e.reduce((a, v) => a + (v - mean) ** 2, 0) / e.length));
  return { w, h, len: g.w, e: e.map((v) => (v - mean) / sd) };
}

// How different two fingerprints are: 0 the same, about 1 unrelated (1 - the best correlation). Lines
// more than 15 % apart in length are different text. b is stretched to a's width first: the detector
// finds the same line a few px taller or shorter from one frame to the next, which changes how wide
// its fingerprint comes out. Then shifts of up to 2 px along and 1 across.
export function printDiff(a, b) {
  if (Math.abs(a.len - b.len) > 0.15 * Math.max(a.len, b.len)) return 1;
  const { w, h } = a, bs = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const u = (x * (b.w - 1)) / Math.max(1, w - 1), x0 = Math.floor(u), x1 = Math.min(b.w - 1, x0 + 1);
      bs[y * w + x] = b.e[y * b.w + x0] + (b.e[y * b.w + x1] - b.e[y * b.w + x0]) * (u - x0);
    }
  }
  let best = -1;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      let sum = 0, n = 0;
      for (let y = Math.max(0, -dy); y < Math.min(h, h - dy); y++) for (let x = Math.max(0, -dx); x < Math.min(w, w - dx); x++, n++) sum += a.e[y * w + x] * bs[(y + dy) * w + x + dx];
      best = Math.max(best, sum / Math.max(1, n));
    }
  }
  return 1 - best;
}

// Greedy CTC decode. Index 0 is the blank, i maps to chars[i - 1], the last class is a space.
export function ctcDecode(probs, n, T, C, chars) {
  const out = [];
  for (let b = 0; b < n; b++) {
    let text = "", prev = 0, sum = 0;
    const steps = [];
    for (let t = 0; t < T; t++) {
      const base = (b * T + t) * C;
      let k = 0, best = probs[base];
      for (let c = 1; c < C; c++) if (probs[base + c] > best) { best = probs[base + c]; k = c; }
      if (k !== 0 && k !== prev) {
        const ch = chars[k - 1] ?? " ";
        text += ch;
        steps.push({ c: ch, t });
        sum += best;
      }
      prev = k;
    }
    out.push({ text, score: steps.length ? sum / steps.length : 0, steps });
  }
  return out;
}

// ---------- full pipeline ----------

// ort: the onnxruntime-web module; det/rec: InferenceSessions; img: {data: RGBA, width, height}.
// Boxes thinner than minLineH px are not recognised (too small to label) and come back in `skipped`;
// `boxes` is every box the detector found, recognised or not (a line's `index` is its place there).
// stop() is checked between model runs; once it returns true the read gives up with a "cancelled"
// error (the shutter does this to a live read still in progress).
export async function runOCR(ort, { det, rec, chars }, img, { detLimit = 960, minLineH = 0, cols = REC_COLS, dropScore = 0.5, stop = () => false } = {}) {
  const check = () => { if (stop()) throw Object.assign(new Error("cancelled"), { cancelled: true }); };
  check();
  const t0 = performance.now();
  const di = detInput(img, detLimit);
  const detOut = await det.run({ [det.inputNames[0]]: new ort.Tensor("float32", di.data, [1, 3, di.h, di.w]) });
  const prob = detOut[det.outputNames[0]].data;
  const boxes = detBoxes(prob, di.w, di.h, di.sx, di.sy);
  const t1 = performance.now();
  check();

  const geoms = boxes.map(cropGeometry);
  const ratio = (i) => geoms[i].w / geoms[i].h;
  const order = geoms.map((g, i) => i).filter((i) => geoms[i].h >= minLineH).sort((a, b) => ratio(a) - ratio(b));
  const recs = new Array(boxes.length);
  for (let s = 0, e; s < order.length; s = e) {
    check();
    // Sorted by ratio, so the crop being added is always the widest in its batch.
    for (e = s + 1; e < order.length && (e - s + 1) * recWidth(ratio(order[e])) <= cols; e++);
    const idx = order.slice(s, e);
    const ri = recInput(img, idx.map((i) => geoms[i]));
    const out = (await rec.run({ [rec.inputNames[0]]: new ort.Tensor("float32", ri.data, [idx.length, 3, REC_H, ri.W]) }))[rec.outputNames[0]];
    const [, T, C] = out.dims;
    ctcDecode(out.data, idx.length, T, C, chars).forEach((r, j) => {
      // Character centres back in image coordinates, from each CTC step's position along the crop.
      const g = geoms[idx[j]];
      r.chars = r.steps.map(({ c, t }) => {
        const f = Math.min(1, ((t + 0.5) * ri.W) / T / ri.widths[j]);
        return { c, x: g.O[0] + f * g.A[0] + 0.5 * g.B[0], y: g.O[1] + f * g.A[1] + 0.5 * g.B[1] };
      });
      recs[idx[j]] = r;
    });
  }
  const t2 = performance.now();

  const lines = [], skipped = [];
  boxes.forEach((pts, i) => {
    const r = recs[i];
    if (!r) return skipped.push(pts);
    if (r.score < dropScore || !r.text.trim()) return;
    const g = geoms[i];
    lines.push({ text: r.text, score: r.score, box: pts, index: i, chars: r.chars, A: g.A, B: g.B, vertical: g.vertical });
  });
  return { lines, skipped, boxes, recognised: order.length, ms: { det: t1 - t0, rec: t2 - t1 }, detSize: [di.w, di.h] };
}
