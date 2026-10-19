// Classic worker (kuromoji is UMD-only): turns recognised lines into pinyin / furigana+romaji / romanization.
importScripts("vendor/pinyin-pro.js", "vendor/wanakana.min.js", "reading.js");

// "file.js:line" of the first stack frame, so errors on a phone say where they came from.
const where = (e) => (String(e?.stack ?? "").match(/([\w.-]+\.m?js):(\d+)/) ?? []).slice(1).join(":");

let japanese = null;

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
      japanese = Reading.initJapanese("vendor/kuromoji/dict/");
    }
    if (lang === "ja") await japanese;
    if (type === "load") return postMessage({ id });
    postMessage({ id, lines: data.texts.map((t) => annotate(lang, t)) });
  } catch (e) {
    if (lang === "ja") japanese = null;
    postMessage({ id, error: String(e?.message ?? e), where: where(e) });
  }
};
