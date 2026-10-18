// Classic worker (kuromoji is UMD-only): turns recognised lines into pinyin / furigana+romaji / romanization.
importScripts("vendor/pinyin-pro.js", "vendor/wanakana.min.js", "reading.js");

let japanese = null;

onmessage = async ({ data }) => {
  const { id, type, lang } = data;
  try {
    if (lang === "ja" && !japanese) {
      importScripts("vendor/kuromoji/kuromoji.js");
      japanese = Reading.initJapanese("vendor/kuromoji/dict/");
    }
    if (lang === "ja") await japanese;
    if (type === "load") return postMessage({ id });
    postMessage({ id, lines: data.texts.map((t) => Reading[lang](t)) });
  } catch (e) {
    if (lang === "ja") japanese = null;
    postMessage({ id, error: String(e?.message ?? e) });
  }
};
