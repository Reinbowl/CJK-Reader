// Write docs/packs.json: the files of each language pack, with sizes (the settings panel's estimates)
// and sha256 (checked by docs/packs.js before a file is cached). Run after vendor and prepare_models.
//   npm run packs
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, writeFileSync } from "node:fs";

const docs = new URL("../docs/", import.meta.url);
// onnxruntime-web's plain WASM build, for marian-worker.js (the OCR may be using the WebGPU one).
const ort = ["vendor/ort/ort.wasm.min.mjs", "vendor/ort/ort-wasm-simd-threaded.mjs", "vendor/ort/ort-wasm-simd-threaded.wasm"];
const pair = (p) => readdirSync(new URL(`models/translate/${p}/`, docs)).filter((f) => f !== "NOTICE").sort().map((f) => `models/translate/${p}/${f}`);
const kuromoji = readdirSync(new URL("vendor/kuromoji/dict/", docs)).sort().map((f) => `vendor/kuromoji/dict/${f}`);

// Reading: what each language adds to the detector and the ORT runtime, which every language needs.
// Japanese reads with the Chinese recogniser (always installed), so its pack is kuromoji alone.
// Translation: a Marian model on onnxruntime (marian-worker.js): encoder, decoder, tokenizer, and ORT's
// WASM build, which the three packs share.
const PACKS = {
  zh: { read: ["models/det.int8.onnx", "models/rec_zhja.int8.onnx", "models/dict_zhja.txt"], translate: [...pair("zh-en"), ...ort] },
  ja: { read: ["vendor/kuromoji/kuromoji.js", ...kuromoji], translate: [...pair("ja-en"), ...ort] },
  ko: { read: ["models/rec_ko.int8.onnx", "models/dict_ko.txt"], translate: [...pair("ko-en"), ...ort] },
};

const files = {};
const entry = (path) => {
  if (!files[path]) {
    const buf = readFileSync(new URL(path, docs));
    files[path] = { path, size: buf.length, sha256: createHash("sha256").update(buf).digest("hex") };
  }
  return files[path];
};
const packs = {};
for (const [lang, kinds] of Object.entries(PACKS)) {
  packs[lang] = {};
  for (const [kind, paths] of Object.entries(kinds)) packs[lang][kind] = paths.map(entry);
}
// core: files the app needs whatever is installed (the OCR runtime), never removed with a pack.
const core = readdirSync(new URL("vendor/ort/", docs)).sort().map((f) => `vendor/ort/${f}`);
// One file per line, so a diff shows which files changed.
const json = JSON.stringify({ core, packs }, null, 1).replace(/\{\s+("path": [^\n]+)\s+("size": [^\n]+)\s+("sha256": [^\n]+)\s+\}/g, "{ $1 $2 $3 }");
writeFileSync(new URL("packs.json", docs), json + "\n");
for (const [lang, kinds] of Object.entries(packs))
  console.log(lang, Object.entries(kinds).map(([k, f]) => `${k} ${(f.reduce((a, e) => a + e.size, 0) / 1e6).toFixed(1)} MB`).join(", "));
