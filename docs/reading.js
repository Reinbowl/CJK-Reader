// Pronunciation for recognised text: pinyin (zh), furigana + romaji (ja), Revised Romanization (ko).
// Classic script: loaded with importScripts in reading-worker.js, or require()d by the Node tests.
// Needs globals pinyinPro and wanakana; kuromoji only for Japanese.
//
// Each function returns { text, units, roman }:
//   units: [{ i, n, parts: [{ s, rt? }], roman }]  i/n index code points of text
//   parts: the unit's text split into ruby pieces (rt = reading shown above s)
//   roman: the whole line romanised
(function (root) {
  "use strict";

  const isKanji = (c) => /[㐀-鿿豈-﫿\u{20000}-\u{3134f}]/u.test(c);
  const isKana = (c) => /[ぁ-ゖァ-ヺー]/.test(c);

  // The recogniser confuses 一 (one) with ー (long-vowel mark). ー only lengthens kana: next to no
  // kana at all it must be 一, and in Japanese it can't follow a kanji, digit, ン or ッ before a kanji.
  function fixDash(chars, lang) {
    return chars.map((c, k) => {
      if (c !== "ー") return c;
      const prev = chars[k - 1] ?? "", next = chars[k + 1] ?? "";
      if (lang === "zh") return isKana(prev) || isKana(next) ? c : "一";
      const lengthenable = isKana(prev) && !"ンッんっー".includes(prev);
      return isKanji(next) && !lengthenable ? "一" : c;
    });
  }

  // ---------- Chinese ----------
  function zh(text) {
    const chars = fixDash(Array.from(text), "zh");
    const all = root.pinyinPro.pinyin(chars.join(""), { type: "all" });
    const units = [];
    all.forEach((p, i) => {
      const last = units[units.length - 1];
      if (p.isZh) units.push({ i, n: 1, parts: [{ s: p.origin, rt: p.pinyin }], roman: p.pinyin });
      else if (last && !last.parts[0].rt) { last.n++; last.parts[0].s += p.origin; last.roman += p.origin; }
      else units.push({ i, n: 1, parts: [{ s: p.origin }], roman: p.origin });
    });
    const roman = units.map((u) => u.roman.trim()).filter(Boolean).join(" ");
    return { text: chars.join(""), units, roman };
  }

  // ---------- Japanese ----------
  let tokenizer = null;
  function initJapanese(dicPath) {
    return new Promise((resolve, reject) => {
      root.kuromoji.builder({ dicPath }).build((err, t) => {
        if (err) return reject(err);
        tokenizer = t;
        resolve();
      });
    });
  }

  // Furigana pieces for one token: strip the kana the surface and reading share at either end,
  // so 食べる/たべる gives 食(た) + べる and お茶/おちゃ gives お + 茶(ちゃ).
  function furigana(surface, reading) {
    const s = Array.from(surface);
    if (!reading || !s.some(isKanji)) return [{ s: surface }];
    const hs = Array.from(root.wanakana.toHiragana(surface, { passRomaji: true }));
    const r = Array.from(root.wanakana.toHiragana(reading));
    let a = 0, b = 0;
    while (a < s.length && a < r.length && !isKanji(s[a]) && hs[a] === r[a]) a++;
    while (b < s.length - a && b < r.length - a && !isKanji(s[s.length - 1 - b]) && hs[hs.length - 1 - b] === r[r.length - 1 - b]) b++;
    const parts = [];
    if (a) parts.push({ s: s.slice(0, a).join("") });
    parts.push({ s: s.slice(a, s.length - b).join(""), rt: r.slice(a, r.length - b).join("") });
    if (b) parts.push({ s: s.slice(s.length - b).join("") });
    return parts;
  }

  // Katakana for romaji: kuromoji's pronunciation marks long vowels (トーキョー, particle は = ワ),
  // but writes エイ as エー too; keep the spelling's イ there so 先生 stays "sensei".
  function spoken(t) {
    const reading = t.reading && t.reading !== "*" ? t.reading : null;
    const pron = t.pronunciation && t.pronunciation !== "*" ? t.pronunciation : null;
    if (!pron) return reading ?? (Array.from(t.surface_form).every(isKana) ? t.surface_form : null);
    let out = pron.replace(/ヲ/g, "オ");
    if (reading && reading.length === pron.length) {
      out = Array.from(out).map((c, k) => (c === "ー" && reading[k] === "イ" ? "イ" : c)).join("");
    }
    return out;
  }

  const MACRON = { a: "ā", i: "ī", u: "ū", e: "ē", o: "ō" };
  function romaji(kana) {
    const chunks = kana.split("ー").map((c) => root.wanakana.toRomaji(c));
    return chunks.map((c, k) => (k < chunks.length - 1 ? c.replace(/[aiueo]$/, (v) => MACRON[v]) : c)).join("");
  }

  // Tokens that attach to the previous word when romanising: auxiliaries after verbs/adjectives
  // and the conjunctive て/で. Keeps 降って as "futte" rather than "fu tte".
  function attaches(t, prev) {
    if (!prev) return false;
    if (t.pos === "助動詞") return ["動詞", "形容詞", "助動詞"].includes(prev.pos);
    return t.pos === "助詞" && t.pos_detail_1 === "接続助詞" && (prev.pos === "動詞" || prev.pos === "形容詞");
  }

  function ja(text) {
    const chars = fixDash(Array.from(text), "ja");
    const fixed = chars.join("");
    if (!tokenizer) return { text: fixed, units: [{ i: 0, n: chars.length, parts: [{ s: fixed }], roman: "" }], roman: "" };
    const units = [];
    let i = 0, prev = null;
    for (let t of tokenizer.tokenize(fixed)) {
      // Signs are short, and with no context IPADIC likes surname readings (出口 = Ideguchi).
      // Re-read a "person name" with a particle after it; a common noun wins if one exists.
      if (t.pos_detail_2 === "人名") {
        const alt = tokenizer.tokenize(t.surface_form + "は")[0];
        if (alt.surface_form === t.surface_form) t = { ...alt, word_position: t.word_position };
      }
      const n = Array.from(t.surface_form).length;
      const kana = spoken(t);
      const parts = furigana(t.surface_form, t.reading && t.reading !== "*" ? t.reading : null);
      const last = units[units.length - 1];
      if (last && attaches(t, prev) && kana != null && last.kana != null) {
        last.n += n;
        last.parts.push(...parts);
        last.kana += kana;
      } else {
        units.push({ i, n, parts, kana });
      }
      i += n;
      prev = t;
    }
    for (const u of units) {
      u.roman = u.kana != null ? romaji(u.kana) : u.parts.map((p) => p.s).join("");
      delete u.kana;
    }
    const roman = units.map((u) => u.roman.trim()).filter(Boolean).join(" ").replace(/ ([、。！？!?,.])/g, "$1");
    return { text: fixed, units, roman };
  }

  // ---------- Korean (Revised Romanization, with the common sound changes) ----------
  const L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ";
  const T = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"];
  const V = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"];
  const INITIAL = { ㄱ: "g", ㄲ: "kk", ㄴ: "n", ㄷ: "d", ㄸ: "tt", ㄹ: "r", ㅁ: "m", ㅂ: "b", ㅃ: "pp", ㅅ: "s", ㅆ: "ss", ㅇ: "", ㅈ: "j", ㅉ: "jj", ㅊ: "ch", ㅋ: "k", ㅌ: "t", ㅍ: "p", ㅎ: "h" };
  // How each final is pronounced at the end of a syllable.
  const FINAL = { "": "", ㄱ: "k", ㄲ: "k", ㅋ: "k", ㄳ: "k", ㄺ: "k", ㄴ: "n", ㄵ: "n", ㄶ: "n", ㄷ: "t", ㅅ: "t", ㅆ: "t", ㅈ: "t", ㅊ: "t", ㅌ: "t", ㅎ: "t", ㄹ: "l", ㄼ: "l", ㄽ: "l", ㄾ: "l", ㅀ: "l", ㅁ: "m", ㄻ: "m", ㅂ: "p", ㅍ: "p", ㅄ: "p", ㄿ: "p", ㅇ: "ng" };
  // Liaison: a final moves onto a following silent ㅇ (clusters split, ㅎ goes silent).
  const LIAISON = { ㄳ: ["ㄱ", "ㅅ"], ㄵ: ["ㄴ", "ㅈ"], ㄶ: ["", "ㄴ"], ㄺ: ["ㄹ", "ㄱ"], ㄻ: ["ㄹ", "ㅁ"], ㄼ: ["ㄹ", "ㅂ"], ㄽ: ["ㄹ", "ㅅ"], ㄾ: ["ㄹ", "ㅌ"], ㄿ: ["ㄹ", "ㅍ"], ㅀ: ["", "ㄹ"], ㅄ: ["ㅂ", "ㅅ"], ㅎ: ["", "ㅇ"] };
  const ASPIRATE = { k: "ㅋ", t: "ㅌ", p: "ㅍ" };
  const NASAL = { k: "ㅇ", t: "ㄴ", p: "ㅁ" };

  function soundChange(a, b) {
    if (b.I === "ㅇ" && a.F && a.F !== "ㅇ") {
      const [stay, move] = LIAISON[a.F] ?? ["", a.F];
      a.F = stay;
      b.I = move;
      if (b.V === 20 && b.I === "ㄷ") b.I = "ㅈ"; // 굳이 guji
      if (b.V === 20 && b.I === "ㅌ") b.I = "ㅊ"; // 같이 gachi
      return;
    }
    if (b.I === "ㅎ" && "ktp".includes(FINAL[a.F]) && FINAL[a.F]) {
      const stay = { ㄺ: "ㄹ", ㄼ: "ㄹ", ㄵ: "ㄴ" }[a.F] ?? "";
      b.I = a.F === "ㅈ" || a.F === "ㄵ" || (a.F === "ㄷ" && b.V === 20) ? "ㅊ" : ASPIRATE[FINAL[a.F]];
      a.F = stay;
      return;
    }
    if (a.F === "ㅎ" || a.F === "ㄶ" || a.F === "ㅀ") {
      const stay = a.F === "ㄶ" ? "ㄴ" : a.F === "ㅀ" ? "ㄹ" : "";
      const asp = { ㄱ: "ㅋ", ㄷ: "ㅌ", ㅈ: "ㅊ", ㅅ: "ㅆ" }[b.I];
      if (asp) { b.I = asp; a.F = stay; return; }
      if (b.I === "ㄴ") a.F = stay || "ㄴ";
    }
    let f = FINAL[a.F];
    if (b.I === "ㄹ") {
      if (f === "n") a.F = "ㄹ"; // 신라 silla
      else if (f && f !== "l") b.I = "ㄴ"; // 종로 jongno, 국립 -> 국닙
    } else if (b.I === "ㄴ" && f === "l") {
      b.I = "ㄹ"; // 설날 seollal
    }
    f = FINAL[a.F];
    if ((b.I === "ㄴ" || b.I === "ㅁ") && NASAL[f]) a.F = NASAL[f]; // 합니다 hamnida, 국물 gungmul
  }

  function ko(text) {
    const chars = Array.from(text);
    const syl = chars.map((c) => {
      const k = c.codePointAt(0) - 0xac00;
      return k >= 0 && k < 11172 ? { I: L[(k / 588) | 0], V: ((k % 588) / 28) | 0, F: T[k % 28] } : null;
    });
    for (let k = 0; k + 1 < syl.length; k++) if (syl[k] && syl[k + 1]) soundChange(syl[k], syl[k + 1]);
    const units = chars.map((c, i) => {
      const s = syl[i];
      if (!s) return { i, n: 1, parts: [{ s: c }], roman: c };
      const initial = s.I === "ㄹ" && FINAL[syl[i - 1]?.F] === "l" ? "l" : INITIAL[s.I];
      const r = initial + V[s.V] + FINAL[s.F];
      return { i, n: 1, parts: [{ s: c, rt: r }], roman: r };
    });
    return { text, units, roman: units.map((u) => u.roman).join("") };
  }

  const api = { zh, ja, ko, initJapanese };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.Reading = api;
})(typeof self !== "undefined" ? self : globalThis);
