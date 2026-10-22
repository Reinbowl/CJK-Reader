// Classic worker (kuromoji is UMD-only): turns recognised lines into pinyin / furigana+romaji / romanization.
importScripts("vendor/pinyin-pro.js", "vendor/wanakana.min.js", "reading.js");

// "file.js:line" of the first stack frame, so errors on a phone say where they came from.
const where = (e) => (String(e?.stack ?? "").match(/([\w.-]+\.m?js):(\d+)/) ?? []).slice(1).join(":");

let japanese = null;

// The Japanese dictionary (12 files, 18 MB): fetched here first with progress for the app's progress
// bar, so kuromoji's own silent requests then come from the service worker's cache.
const DICT = ["base", "cc", "check", "tid", "tid_map", "tid_pos", "unk", "unk_char", "unk_compat", "unk_invoke", "unk_map", "unk_pos"];
async function fetchDictionary() {
  const got = {}, size = {};
  let last = 0;
  const report = (end) => {
    if (!end && performance.now() - last < 100) return;
    last = performance.now();
    const loaded = Object.values(got).reduce((a, b) => a + b, 0);
    postMessage({ type: "progress", file: "dictionary", loaded, total: end ? loaded : Object.values(size).reduce((a, b) => a + b, 0) });
  };
  await Promise.all(DICT.map(async (name) => {
    const res = await fetch(`vendor/kuromoji/dict/${name}.dat.gz`);
    if (!res.ok) throw new Error(`${name}.dat.gz: HTTP ${res.status}`);
    size[name] = Number(res.headers.get("content-length")) || 0;
    got[name] = 0;
    const reader = res.body.getReader();
    for (let r; !(r = await reader.read()).done;) {
      got[name] += r.value.length;
      report(false);
    }
  }));
  report(true);
}

// Live mode sends the same lines again and again: keep the last readings (least recently used out).
const memo = new Map();
function annotate(lang, text) {
  const key = `${lang}\n${text}`;
  let r = memo.get(key);
  if (r) memo.delete(key);
  else {
    r = Reading[lang](text);
    if (memo.size >= 1000) memo.delete(memo.keys().next().value);
  }
  memo.set(key, r);
  return r;
}

onmessage = async ({ data }) => {
  const { id, type, lang } = data;
  try {
    if (lang === "ja" && !japanese) {
      importScripts("vendor/kuromoji/kuromoji.js");
      japanese = fetchDictionary().then(() => Reading.initJapanese("vendor/kuromoji/dict/"));
    }
    if (lang === "ja") await japanese;
    if (type === "load") return postMessage({ id });
    postMessage({ id, lines: data.texts.map((t) => annotate(lang, t)) });
  } catch (e) {
    if (lang === "ja") japanese = null;
    postMessage({ id, error: String(e?.message ?? e), where: where(e) });
  }
};
