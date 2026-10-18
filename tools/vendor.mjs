// Copy the runtime files the site needs from node_modules into docs/vendor (served by GitHub Pages).
//   npm run vendor
import { cpSync, mkdirSync, readdirSync, rmSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const nm = new URL("../node_modules/", import.meta.url);
const out = new URL("../docs/vendor/", import.meta.url);
rmSync(out, { recursive: true, force: true });

const copy = (from, to) => {
  mkdirSync(new URL(".", new URL(to, out)), { recursive: true });
  cpSync(new URL(from, nm), new URL(to, out), { recursive: true });
};

// onnxruntime-web: WASM build (default) and WebGPU build (?ep=webgpu), each with its own .wasm.
for (const f of [
  "ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm",
  "ort.webgpu.min.mjs", "ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm",
]) copy(`onnxruntime-web/dist/${f}`, `ort/${f}`);

copy("pinyin-pro/dist/index.js", "pinyin-pro.js");
copy("pinyin-pro/LICENSE", "pinyin-pro.LICENSE");
copy("wanakana/wanakana.min.js", "wanakana.min.js");
copy("wanakana/LICENSE", "wanakana.LICENSE");
copy("kuromoji/build/kuromoji.js", "kuromoji/kuromoji.js");
copy("kuromoji/dict", "kuromoji/dict");
copy("kuromoji/LICENSE-2.0.txt", "kuromoji/LICENSE-2.0.txt");
copy("kuromoji/NOTICE.md", "kuromoji/NOTICE.md");

const size = (p) => statSync(p).isDirectory() ? readdirSync(p).reduce((n, f) => n + size(join(p, f)), 0) : statSync(p).size;
console.log(`docs/vendor: ${(size(fileURLToPath(out)) / 1e6).toFixed(1)} MB`);
