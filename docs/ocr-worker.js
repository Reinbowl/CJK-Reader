// Module worker: loads onnxruntime-web and the OCR models, runs OCR on frames from app.js.
import { runOCR } from "./ocr-core.js";

// "file.js:line" of the first stack frame, so errors on a phone say where they came from.
const where = (e) => (String(e?.stack ?? "").match(/([\w.-]+\.m?js):(\d+)/) ?? []).slice(1).join(":");

const DICT = { zhja: "dict_zhja.txt", ko: "dict_ko.txt" };
let ort, det, opts, ready;
let backend, fallback = null, gpu = null; // the backend actually in use, and why it isn't the one asked for
const engines = {}, models = {}; // models: downloaded bytes, kept to rebuild sessions on WASM

// Everything that runs or creates a session goes through here, one at a time. Messages keep arriving
// while a read is in progress (a snap during a live read, a language switch), and interleaved runs
// break the WebGPU backend ("Failed to download data from buffer"); on WASM they gain nothing.
let queue = Promise.resolve();
function serial(fn) {
  const run = queue.then(fn);
  queue = run.catch(() => {});
  return run;
}
let cancelBefore = 0; // reads sent before the last "cancel" give up at their next check

// Fetch a file, telling the app how far along it is ("progress" events, for the progress bar).
async function download(url, file) {
  const res = await fetch(url);
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

const bytes = (file) => (models[file] ??= download(new URL(`models/${file}`, import.meta.url), file).catch((e) => { delete models[file]; throw e; }));

// One backend per session, no ["webgpu", "wasm"] list: ORT would fall back silently, and the status
// bar would claim WebGPU while WASM does the work.
async function session(file) {
  const model = await bytes(file);
  const create = () => serial(() => ort.InferenceSession.create(model, {
    executionProviders: [backend],
    graphOptimizationLevel: "all",
    logSeverityLevel: 3, // errors only, not the "nodes not assigned to the preferred provider" notice
    // ORT's threads busy-wait between ops by default: no faster here, and it heats the phone.
    extra: { session: { intra_op: { allow_spinning: opts.spin ? "1" : "0" } } },
  }));
  if (backend !== "webgpu") return create();
  try {
    return await create();
  } catch (e) {
    fallback = `WebGPU failed: ${e?.message ?? e}`;
    backend = "wasm";
    return create();
  }
}

// After a WebGPU run fails anyway (say the GPU device was lost): rebuild every session on WASM, tell
// the app, and carry on there until the next launch.
let rebuilding = null;
function toWasm(reason) {
  rebuilding ??= (async () => {
    fallback = `WebGPU failed: ${reason}`;
    backend = "wasm";
    for (const m in engines) delete engines[m];
    det = await session(`det.${opts.precision}.onnx`);
    postMessage({ type: "backend", backend, fallback, gpu });
  })();
  return rebuilding;
}

function engine(model) {
  engines[model] ??= (async () => {
    await ready;
    await rebuilding;
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
        backend = data.ep;
        if (backend === "webgpu") {
          const adapter = await navigator.gpu?.requestAdapter();
          if (adapter) gpu = [adapter.info?.vendor, adapter.info?.architecture, adapter.features.has("shader-f16") && "f16"].filter(Boolean).join(" ") || "yes";
          else { fallback = navigator.gpu ? "no WebGPU adapter" : "no WebGPU in this browser"; backend = "wasm"; }
        }
        // Start everything the first read needs at once, so the app can show one progress bar: the
        // runtime's .wasm (fetched here and handed to ORT, which would otherwise fetch it silently),
        // the detector, and the recogniser for the current language. The WebGPU build (27 MB) also
        // runs WASM sessions, for a fallback after a failed WebGPU one.
        const runtime = backend === "webgpu" ? "ort-wasm-simd-threaded.asyncify.wasm" : "ort-wasm-simd-threaded.wasm";
        const wasm = download(new URL(`vendor/ort/${runtime}`, import.meta.url), runtime);
        bytes(`det.${data.precision}.onnx`).catch(() => {});
        if (data.model) bytes(`rec_${data.model}.${data.precision}.onnx`).catch(() => {});
        ort = await import(backend === "webgpu" ? "./vendor/ort/ort.webgpu.min.mjs" : "./vendor/ort/ort.wasm.min.mjs");
        ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;
        ort.env.wasm.numThreads = data.threads;
        ort.env.wasm.wasmBinary = await wasm;
        det = await session(`det.${data.precision}.onnx`);
      })();
      await ready;
      postMessage({ id, backend, fallback, gpu });
    } else if (type === "load") {
      await engine(data.model);
      postMessage({ id });
    } else if (type === "ocr") {
      const img = { data: new Uint8ClampedArray(data.buffer), width: data.width, height: data.height };
      const read = async () => {
        const e = await engine(data.model);
        return serial(() => runOCR(ort, e, img, { detLimit: data.detLimit, minLineH: data.minLineH, stop: () => id < cancelBefore }));
      };
      let res;
      try {
        res = await read();
      } catch (e) {
        if (e?.cancelled || (backend !== "webgpu" && !rebuilding)) throw e;
        await toWasm(String(e?.message ?? e));
        res = await read();
      }
      postMessage({ id, ...res });
    } else if (type === "cancel") {
      cancelBefore = id;
      postMessage({ id });
    }
  } catch (e) {
    postMessage({ id, error: String(e?.message ?? e), where: where(e), cancelled: !!e?.cancelled });
  }
};
