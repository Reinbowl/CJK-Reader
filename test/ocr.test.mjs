// Runs docs/ocr-core.js through onnxruntime-web (WASM, as in the browser) on test/images
// and compares with the ground truth and the Python reference (test/ref_int8.json).
//   node test/ocr.test.mjs [int8|fp32]
import { readFileSync, existsSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { PNG } from "pngjs";
import { detBoxes, runOCR } from "../docs/ocr-core.js";

const root = new URL("../", import.meta.url);
const precision = process.argv[2] ?? "int8";
const truth = JSON.parse(readFileSync(new URL("test/images/truth.json", root), "utf8"));
const refPath = new URL("test/ref_int8.json", root);
const ref = existsSync(refPath) ? JSON.parse(readFileSync(refPath, "utf8")) : {};

ort.env.wasm.numThreads = 1;
const session = (name) => ort.InferenceSession.create(readFileSync(new URL(`docs/models/${name}.${precision}.onnx`, root)));
const dict = (name) => readFileSync(new URL(`docs/models/${name}`, root), "utf8").replace(/\n$/, "").split("\n");

const det = await session("det");
const engines = {
  zhja: { det, rec: await session("rec_zhja"), chars: dict("dict_zhja.txt") },
  ko: { det, rec: await session("rec_ko"), chars: dict("dict_ko.txt") },
};

function cer(a, b) {
  a = [...a]; b = [...b];
  let d = [...Array(b.length + 1).keys()];
  for (let i = 1; i <= a.length; i++) {
    const nd = [i];
    for (let j = 1; j <= b.length; j++) nd[j] = Math.min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (a[i - 1] !== b[j - 1]));
    d = nd;
  }
  return d[b.length] / Math.max(1, a.length);
}

let failed = false;

// Camera noise: isolated pixels and 1-px lines above threshold must be skipped, not crash.
{
  const w = 64, h = 32, prob = new Float32Array(w * h);
  prob[10 * w + 20] = 0.9;
  prob.fill(0.9, 20 * w + 5, 20 * w + 15);
  const boxes = detBoxes(prob, w, h, 1, 1);
  console.log(`noise blobs -> ${boxes.length} boxes`);
  if (boxes.length) failed = true;
}
for (const [name, t] of Object.entries(truth)) {
  const png = PNG.sync.read(readFileSync(new URL(`test/images/${name}.png`, root)));
  const img = { data: png.data, width: png.width, height: png.height };
  const { lines, ms } = await runOCR(ort, engines[t.lang === "ko" ? "ko" : "zhja"], img, { detLimit: 960 });
  const got = lines.map((l) => l.text);
  const err = t.lines.map((s) => Math.min(1, ...got.map((g) => cer(s, g))));
  const meanCer = err.reduce((a, b) => a + b, 0) / err.length;
  const refTexts = (ref[name] ?? []).map((r) => r.text);
  const sameAsRef = refTexts.length === got.length && refTexts.every((r, i) => r === got[i]);
  console.log(`${name.padEnd(8)} lines ${got.length}/${t.lines.length}  CER ${meanCer.toFixed(3)}  det ${ms.det.toFixed(0)}ms rec ${ms.rec.toFixed(0)}ms  ${sameAsRef ? "== python" : "!= python"}`);
  console.log("   ", got.join(" | "));
  if (!sameAsRef) console.log("    python:", refTexts.join(" | "));
  // Character positions must sit inside the line's box.
  for (const l of lines) {
    const xs = l.box.map((p) => p[0]), ys = l.box.map((p) => p[1]);
    for (const c of l.chars) {
      if (c.x < Math.min(...xs) - 1 || c.x > Math.max(...xs) + 1 || c.y < Math.min(...ys) - 1 || c.y > Math.max(...ys) + 1) {
        console.log("    char outside box:", l.text, c);
        failed = true;
      }
    }
  }
  if (meanCer > 0.1) failed = true;
}
process.exit(failed ? 1 : 0);
