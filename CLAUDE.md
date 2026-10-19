# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CJK Reader is an offline camera PWA. It runs OCR on Chinese, Japanese or Korean text and shows how to read it (pinyin, furigana and romaji, or Revised Romanization). Everything runs in the browser. The README covers model sizes, URL switches for performance testing, and known limitations.

## Commands

```sh
npm ci                                  # JS deps (exact versions pinned in package.json)
npm run vendor                          # wipe and rebuild docs/vendor from node_modules
uv run tools/prepare_models.py [--fp32] # download pinned models, verify sha256, int8-quantize -> docs/models
npm test                                # both Node test suites
node test/ocr.test.mjs [int8|fp32]      # OCR pipeline only (needs docs/models)
node test/reading.test.cjs              # pinyin / romaji / Korean romanization only
npm run e2e                             # Playwright, headless Chromium, fake camera (~25 s)
npm run serve                           # http://localhost:8080 (camera needs HTTPS or localhost)
uv run tools/ocr_ref.py --json test/ref_int8.json   # regenerate the Python reference output
uv run tools/make_test_images.py --fonts .cache/fonts --scenes calendar  # regenerate one test image
```

**Fonts:** `make_test_images.py` needs Noto CJK OTFs. The calendar scene only needs `NotoSansCJKsc-Regular.otf`, from github.com/notofonts/noto-cjk `Sans/OTF/SimplifiedChinese/`, saved in `.cache/fonts/` (gitignored). The sign scenes share one rng, so regenerating any of them needs all four fonts. `npm run e2e` also points Chromium's fontconfig at `.cache/fonts` when it exists, so screenshots show CJK text instead of tofu.

**`npm run e2e`:**
- Plays Y4M clips that `test/e2e.test.mjs` writes into `.cache/e2e/` (608×1080 portrait; the fake device crops anything taller to the 1080 px the app requests).
- Serves `docs/` without COOP/COEP, so the service worker's reload path is tested.
- Asserts on `self.cjkStats` (set in `app.js`), not on status-bar text.
- Saves screenshots to `.cache/e2e/*.png`. Uses the `playwright` devDependency, pinned to match the Chromium build cached in `~/.cache/ms-playwright`.

There's no build step, bundler, linter or TypeScript. `docs/` is the deployed site as-is (GitHub Pages serves `main` `/docs`). `docs/models/` and `docs/vendor/` are generated, but they're committed because Pages serves them. Don't edit them by hand. `*.fp32.onnx` is gitignored.

When does `ocr.test.mjs` fail? Only when an image's mean CER is over 0.1, a character position falls outside its line box, or the noise-blob or `minLineH` checks fail. Its `!= python` lines are information only. Readings are added to `test/reading.test.cjs` as `[text, expectedRoman]` pairs, and `null` means "must not throw". Every case also checks that `units` tile the text exactly.

## Architecture

```
app.js (main thread) ──rpc──► ocr-worker.js (module worker) → ocr-core.js + onnxruntime-web
                     ──rpc──► reading-worker.js (classic worker) → reading.js + pinyin-pro / wanakana / kuromoji
```

- **RPC:** `rpc()` in `app.js` matches replies by `id`. A worker message without an `id` is an event (download `progress`). A worker error comes back as `{id, error, where}`, where `where` is the `file.js:line` of the first stack frame, shown in the status bar because phones have no console.
- **`ocr-core.js`** is pure JS on RGBA buffers. It has no DOM and takes `ort` as a parameter, so the Node tests run the exact code the browser does. It mirrors PaddleOCR's pre- and post-processing. `tools/ocr_ref.py` is the Python reference (OpenCV and pyclipper), so a change to detection or recognition should still match it. The models expect BGR, and `sampleBGR` swaps the channels.
  - One deliberate difference from PaddleOCR: recognizer crops are padded to `REC_MIN_W` (64), not 320, and batched by a column budget (`REC_COLS`). `ocr_ref.py` `recognize()` mirrors this, so keep the two in step.
  - `runOCR`'s `minLineH` leaves thinner boxes unrecognised and returns them in `skipped`. Live mode passes `MIN_LINE / map.s`, which is derived from `draw()`'s label limits (`LABEL_SCALE`, `MIN_LABEL`).
- **Live loop (`app.js`):** reads are gated, not continuous.
  - `sample()` runs every `GATE_MS` and compares normalised grey thumbnails (`thumbnail()`, `compare()` with a ±3 px shift search), setting `gate.state` to moving, ready or idle.
  - `liveLoop` reads only in `ready`, with a duty-cycle gap (`MIN_GAP`, and at least the last read's duration).
  - `#overlay` gets the `faded` class while moving, and a CSS translate for small drift.
  - The thresholds (`MOVE_*`, `CHANGE_*`) are tuned on a phone with `?debug=1`.
- **`reading.js`** must stay a classic IIFE script. kuromoji is UMD-only, so the reading worker is a classic worker that uses `importScripts`. The Node test `require()`s the same file. It reads `pinyinPro`, `wanakana` and `kuromoji` as globals and exposes `Reading.{zh,ja,ko,initJapanese}`. kuromoji and its 18 MB dictionary load only when Japanese is picked.
- **Position contract between the two halves:** each OCR line has `chars: [{c, x, y}]`, one entry per code point of `text`. Each reading unit `{i, n, parts, roman}` indexes code points of that same text, and `app.js` places a label at `line.chars.slice(u.i, u.i + u.n)`. So reading code must keep the code-point count unchanged. `fixDash` (ー→一) swaps one character for one character.
- **Backend:** `ocr-worker.js` creates each session with exactly one backend, and with `allow_spinning` off unless `?spin=1`. It reports which backend actually ran: WebGPU falls back to WASM with a `fallback` reason rather than silently.
- **iOS bottom strip:** in iOS home-screen mode, `fitScreen()` in `app.js` measures any strip between the layout viewport and the screen bottom into the CSS var `--gap`. `#view`, the footer and the sheet extend over it.
- **Models:** the detector `det.*.onnx` is shared. The recognizer comes from `MODEL` in `app.js`: zh and ja use `rec_zhja`, ko uses `rec_ko`. Each recognizer has a dictionary in which output index `i+1` is line `i`, index 0 is the CTC blank, and the last class is a space. "int8" means weight-only quantization with `DequantizeLinear`, and compute stays fp32, so the same files run on both WASM and WebGPU.
- **Service worker (`sw.js`):**
  - It adds the COOP/COEP headers that WASM threads (SharedArrayBuffer) need, since GitHub Pages can't send them. `app.js` reloads once to become cross-origin isolated, using the sessionStorage key `coi-reload`.
  - Two caches:
    - `SHELL`: app code, network-first with `no-cache`.
    - `ASSETS`: anything under `/models/` or `/vendor/`, cache-first.
  - **Bump `ASSETS` whenever model or vendor files change**, or installed apps keep the old ones.
  - A new top-level app file also needs adding to `SHELL_FILES`.
