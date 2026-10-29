// The ONNX translators (docs/marian.js): the tokenizer must give exactly the ids Hugging Face's
// MarianTokenizer does (test/marian_ref.json, from tools/marian_ref.py), and each model must turn a few
// sentences into recognisable English. Needs docs/models/translate (uv run tools/prepare_models.py).
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { tokenizer, translate } from "../docs/marian.js";

const root = new URL("../", import.meta.url);
const ref = JSON.parse(readFileSync(new URL("test/marian_ref.json", root), "utf8"));
ort.env.wasm.numThreads = 1;

// Sentence -> a pattern its English must match.
const CASES = {
  "ja-en": [["東京駅はこちらです", /Tokyo Station/i], ["今日は雨が降っています", /rain/i], ["机に花があります。", /flower/i],
    ["日本語を勉強するのは楽しいです。毎日少しずつ練習しています。", /Japanese.*every day/i]],
  "ko-en": [["서울역 출구는 왼쪽입니다", /Seoul Station.*left/i], ["한국어를 공부합니다", /Korean/i], ["같이 먹어요", /eat|together/i]],
  "zh-en": [["欢迎光临北京大学", /Welcome.*University/i], ["今天天气很好，我们去公园散步。", /park/i], ["重庆火锅", /Chongqing/i]],
};

let failed = false;
for (const [pair, cases] of Object.entries(CASES)) {
  const dir = new URL(`docs/models/translate/${pair}/`, root);
  const tok = tokenizer(JSON.parse(readFileSync(new URL("tokenizer.json", dir), "utf8")));
  const bad = ref[pair].filter(({ text, ids }) => tok.encode(text).join() !== ids.join());
  console.log(`${pair.padEnd(10)} tokenizer: ${ref[pair].length - bad.length}/${ref[pair].length} as MarianTokenizer`);
  for (const { text, ids } of bad.slice(0, 5)) console.log(`    ${JSON.stringify(text)}\n      want ${ids}\n      got  ${tok.encode(text)}`);
  if (bad.length) failed = true;

  const session = (f) => ort.InferenceSession.create(readFileSync(new URL(f, dir)), { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  const model = { encoder: await session("encoder.int8.onnx"), decoder: await session("decoder.int8.onnx"), tok };
  for (const [text, want] of cases) {
    const t = performance.now();
    const en = await translate(ort, model, text);
    const ok = want.test(en);
    console.log(`  ${ok ? "ok  " : "FAIL"} ${text} => ${en}  (${(performance.now() - t).toFixed(0)} ms)`);
    if (!ok) failed = true;
  }
}
process.exit(failed ? 1 : 0);
