// Module worker: loads onnxruntime-web and the PP-OCRv5 models, runs OCR on frames from app.js.
import { runOCR } from "./ocr-core.js";

// "file.js:line" of the first stack frame, so errors on a phone say where they came from.
const where = (e) => (String(e?.stack ?? "").match(/([\w.-]+\.m?js):(\d+)/) ?? []).slice(1).join(":");

const DICT = { zhja: "dict_zhja.txt", ko: "dict_ko.txt" };
let ort, det, opts, ready;
const engines = {};

async function download(file) {
  const res = await fetch(new URL(`models/${file}`, import.meta.url));
  if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0, last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (performance.now() - last > 100) {
      last = performance.now();
      postMessage({ type: "progress", file, loaded, total });
    }
  }
  postMessage({ type: "progress", file, loaded, total: loaded });
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  return buf;
}

async function session(file) {
  return ort.InferenceSession.create(await download(file), {
    executionProviders: opts.ep === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
    graphOptimizationLevel: "all",
  });
}

function engine(model) {
  engines[model] ??= (async () => {
    await ready;
    const [rec, dict] = await Promise.all([
      session(`rec_${model}.${opts.precision}.onnx`),
      fetch(new URL(`models/${DICT[model]}`, import.meta.url)).then((r) => r.text()),
    ]);
    return { det, rec, chars: dict.replace(/\n$/, "").split("\n") };
  })().catch((e) => { delete engines[model]; throw e; });
  return engines[model];
}

onmessage = async ({ data }) => {
  const { id, type } = data;
  try {
    if (type === "init") {
      opts = data;
      ready = (async () => {
        ort = await import(data.ep === "webgpu" ? "./vendor/ort/ort.webgpu.min.mjs" : "./vendor/ort/ort.wasm.min.mjs");
        ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;
        ort.env.wasm.numThreads = data.threads;
        det = await session(`det.${data.precision}.onnx`);
      })();
      await ready;
      postMessage({ id });
    } else if (type === "load") {
      await engine(data.model);
      postMessage({ id });
    } else if (type === "ocr") {
      const e = await engine(data.model);
      const img = { data: new Uint8ClampedArray(data.buffer), width: data.width, height: data.height };
      postMessage({ id, ...(await runOCR(ort, e, img, { detLimit: data.detLimit })) });
    }
  } catch (e) {
    postMessage({ id, error: String(e?.message ?? e), where: where(e) });
  }
};
