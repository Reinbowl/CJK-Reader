// Module worker: one language's Marian translator (marian.js) on onnxruntime-web's WASM backend, into
// English. app.js starts it after a snap when that language's translation pack is installed and ends it
// on a language switch. Its WASM memory holds the int8 model, about 230 MB for FuguMT.
import { tokenizer, translate } from "./marian.js";

// "file.js:line" of the first stack frame, so errors on a phone say where they came from.
const where = (e) => (String(e?.stack ?? "").match(/([\w.-]+\.m?js):(\d+)/) ?? []).slice(1).join(":");

const get = async (path) => {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path.split("/").pop()}: HTTP ${res.status}`);
  return res;
};

let engine = null;
async function load(files) {
  const ort = await import("./vendor/ort/ort.wasm.min.mjs");
  ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;
  ort.env.wasm.numThreads = 1; // a sentence is a few dozen small steps: threads would only add memory
  const file = (name) => files.find((f) => f.split("/").pop() === name);
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all", logSeverityLevel: 3 };
  // One session at a time: creating one copies its model into WASM memory.
  const session = async (name) => ort.InferenceSession.create(new Uint8Array(await (await get(file(name))).arrayBuffer()), opts);
  const tok = tokenizer(await (await get(file("tokenizer.json"))).json());
  const encoder = await session("encoder.int8.onnx");
  const decoder = await session("decoder.int8.onnx");
  return { ort, model: { encoder, decoder, tok } };
}

// One request at a time: a translation runs the decoder step by step, and the next waits for it.
let queue = Promise.resolve();
onmessage = ({ data }) => {
  const { id, type } = data;
  queue = queue.then(async () => {
    try {
      if (type === "load") {
        engine ??= load(data.files).catch((e) => { engine = null; throw e; });
        await engine;
        postMessage({ id });
      } else if (type === "translate") {
        const { ort, model } = await engine;
        const texts = [];
        for (const t of data.texts) texts.push(await translate(ort, model, t));
        postMessage({ id, texts });
      }
    } catch (e) {
      postMessage({ id, error: String(e?.message ?? e), where: where(e) });
    }
  });
};
