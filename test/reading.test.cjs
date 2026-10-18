// node test/reading.test.cjs
const path = require("node:path");
globalThis.pinyinPro = require("pinyin-pro");
globalThis.wanakana = require("wanakana");
globalThis.kuromoji = require("kuromoji");
require("../docs/reading.js"); // ESM under "type": "module", so it sets globalThis.Reading
const Reading = globalThis.Reading;

const cases = {
  zh: [
    ["银行行长", "yín háng háng zhǎng"],
    ["重庆火锅", "chóng qìng huǒ guō"],
    ["长城", "cháng chéng"],
    ["ラーメンー杯", null], // ー -> 一 in Chinese mode; just must not throw
    ["营业时间 9:00-17:00", "yíng yè shí jiān 9:00-17:00"],
  ],
  ko: [
    ["안녕하세요", "annyeonghaseyo"],
    ["감사합니다", "gamsahamnida"],
    ["한국어", "hangugeo"],
    ["신라", "silla"],
    ["종로", "jongno"],
    ["같이", "gachi"],
    ["먹어요", "meogeoyo"],
    ["왼쪽입니다", "oenjjogimnida"],
    ["좋아요", "joayo"],
    ["좋다", "jota"],
    ["없어요", "eopseoyo"],
    ["설날", "seollal"],
    ["국물", "gungmul"],
    ["독립", "dongnip"],
    ["백마", "baengma"],
    ["학교", "hakgyo"],
    ["놓는", "nonneun"],
    ["않아요", "anayo"],
    ["맛있어요", "masisseoyo"],
    ["읽어요", "ilgeoyo"],
    ["반갑습니다", "bangapseumnida"],
    ["한국어를", "hangugeoreul"],
    ["서울", "seoul"],
    ["영업시간 오전 10시", "yeongeopsigan ojeon 10si"],
  ],
  ja: [
    ["東京駅はこちらです", "tōkyō eki wa kochira desu"],
    ["今日は雨が降っています", "kyō wa ame ga futte imasu"],
    ["ラーメンー杯八百円", null],
    ["先生", "sensei"],
    ["お茶を飲みたい", "ocha o nomitai"],
    ["出口", "deguchi"],
    ["出口改札口", "deguchi kaisatsu guchi"],
    ["田中さん", "tanaka san"],
  ],
};

(async () => {
  await Reading.initJapanese(path.join(__dirname, "../node_modules/kuromoji/dict"));
  let failed = 0;
  for (const [lang, list] of Object.entries(cases)) {
    for (const [text, want] of list) {
      const r = Reading[lang](text);
      // units must tile the text exactly
      const rebuilt = r.units.map((u) => u.parts.map((p) => p.s).join("")).join("");
      const tiles = r.units.every((u, k) => k === 0 || u.i === r.units[k - 1].i + r.units[k - 1].n);
      const ok = (want == null || r.roman === want) && rebuilt === r.text && tiles;
      if (!ok) failed++;
      const ruby = r.units.map((u) => u.parts.map((p) => (p.rt ? `${p.s}(${p.rt})` : p.s)).join("")).join(" ");
      console.log(`${ok ? "ok  " : "FAIL"} ${lang} ${text} -> ${r.roman}${want != null && r.roman !== want ? `   (want ${want})` : ""}\n       ${ruby}`);
    }
  }
  process.exit(failed ? 1 : 0);
})();
