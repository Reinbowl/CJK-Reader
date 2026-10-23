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
npm run e2e                             # Playwright, headless Chromium, fake camera (~1 min)
E2E_WEBGPU=1 npm run e2e                # also a WebGPU scenario on SwiftShader (slow: ~10 s per read)
E2E_ONLY=pan,tremor npm run e2e         # some scenarios (download, static, calendar, carton, stack, page, zoom, pan, tremor, wander, webgpu)
npm run serve                           # http://localhost:8080 (camera needs HTTPS or localhost)
uv run tools/ocr_ref.py --json test/ref_int8.json   # regenerate the Python reference output
uv run tools/make_test_images.py --fonts .cache/fonts --scenes calendar carton page subs  # regenerate these test images
```

**Fonts:** `make_test_images.py` needs Noto CJK OTFs. The calendar, carton, page and subs scenes only need `NotoSansCJKsc-Regular.otf`, from github.com/notofonts/noto-cjk `Sans/OTF/SimplifiedChinese/`, saved in `.cache/fonts/` (gitignored). The sign scenes share one rng, so regenerating any of them needs all four fonts. `npm run e2e` also points Chromium's fontconfig at `.cache/fonts` when it exists, so screenshots show CJK text instead of tofu.

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
  - `runOCR`'s `minLineH` leaves thinner boxes unrecognised and returns them in `skipped`. Live mode passes `MIN_LINE / map.s`, where `MIN_LINE` (screen px, default 14) is a dev-panel setting (`minline`).
- **Live loop (`app.js`):** reads are gated, not continuous.
  - `sample()` runs every `GATE_MS` and compares normalised grey thumbnails (`thumbnail()`, `compare()` with a shift search of ±`reach()` px), setting `gate.state` to moving, ready or idle and `gate.fast`.
  - "Same scene" means the residual after the best shift (sub-pixel: `compare()` samples bilinearly there) is under `2 × gate.noise + 0.1`. `gate.noise` is a running average of that residual while steady, so big sharp text tolerates more jitter.
  - The thumbnail is `THUMB_W / zoom` px wide, down to `MIN_THUMB` (16 px, reached at 2.5×). Past that its shifts are scaled by its `u` (1× thumbnail px per px). The thresholds are in 1× thumbnail px, so they mean the same hand movement at any zoom. The shift search (`reach()`) covers `SEARCH` of those px, but at most a quarter of the thumbnail. The e2e `tremor` scene checks that 2× and 3× are judged like 1×, and `wander` that labels follow a slow drift at 3×.
  - `FAST_PX` (per sample) fades the overlay. Below `STEADY_PX` for `SETTLE` samples the view counts as steady. A read happens once drift since the last read exceeds `CHANGE_PX`, or the scene changes.
  - `liveLoop` reads only in `ready`, with a duty-cycle gap (`MIN_GAP`, and at least the last read's duration).
  - `#overlay` gets the `faded` class on fast moves, and otherwise a CSS translate that follows the drift.
  - Tune the thresholds on a phone with the dev panel's readout, which shows shifts in 1× thumbnail px, like the thresholds. The e2e `pan` (fast), `tremor` (big text, jitter) and `wander` (slow drift) clips pin the behaviour.
- **Zoom (`app.js`):**
  - Live: `zoom` = the camera's own zoom (`hwZoom`, from `getCapabilities().zoom`, applied by `cameraZoom()` with one `applyConstraints` in flight) × digital `dzoom`.
  - Range: up to `CAMERA_MAX` (5) with camera zoom (iOS Safari reports 0.5–10×), `DIGITAL_MAX` (3) without, and `STILL_ZOOM_MAX` (5) for frozen frames.
  - The slider's marks are rebuilt in `applyZoom()` for the current range. `wakeZoombar()` shows it in full for 2 s; at rest it's a line.
  - Digital zoom scales `#cam` with CSS and passes `dzoom` to `visibleRect()`, which `grab()` and `thumbnail()` use, so reads and the motion gate see the same crop.
  - The shutter freezes the visible rectangle as a full-resolution `ImageBitmap`. `lastStill` keeps `src`, `rect`, `k`, `map`, `result` and `plan`.
  - Still zoom: `sv` (screen = `sv.s` × base + `sv.t`). `setStillView()` repaints on the next animation frame: `paintStill()` draws the image from its source, and `paint(lastStill.plan, sv)` draws the labels. No CSS stretching: a stretched canvas only covers the area it was drawn for.
  - `lastStill.plan` is computed once per read with `layout()`. Until the still's read is done, it's the last live plan (`livePlan`). Zooming therefore enlarges boxes and labels like a page and never rearranges them.
  - `sizeCanvas()` only reallocates a canvas when its size changes, because frames repaint during a pinch.
  - Gestures are pointer events on `#view` (`touch-action: none`, Safari `gesturestart` cancelled). Double-tap resets.
  - `layout()` skips lines entirely off screen and only shifts a row back on screen when its line's centre is on screen.
  - The e2e `zoom` scene drives multi-touch through CDP `Input.dispatchTouchEvent`.
- **Downloads:**
  - On `init`, `ocr-worker.js` fetches the runtime `.wasm` itself and hands it over as `ort.env.wasm.wasmBinary`. It downloads the detector and the current language's recognizer in parallel, all through `download()`, which posts `progress` events.
  - `reading-worker.js` does the same for the kuromoji dictionary (`fetchDictionary()`).
  - `onProgress()` in `app.js` sums them into one bar (`#progress`); `showReady()` runs when the models have actually loaded.
  - A failed first `init` is retried by the next `ensureModels()`.
  - The e2e `download` scene throttles `.onnx` and `.wasm` files.
- **Labels (`layout()` / `paint()` in `app.js`):**
  - One row per horizontal line, its font from the line's thickness (`LABEL_SCALE`, at most `MAX_LABEL`) and shrunk so the row fits the line and the screen. There's no minimum size: small print gets small pinyin.
  - Labels are placed by `spread()`: an isotonic-regression least-squares fit, so labels never overlap within a row.
  - Rows are placed biggest line first, lines within 15 % of the same size top to bottom. Each side's biggest clear size is found (`largest()`); above wins unless below is bigger (for small rows, more than 1/0.7 bigger, so a paragraph's rows don't flip sides on sub-pixel gap differences). `overlaps()` is a separating-axis test against other lines' boxes and placed rows.
  - Rows readable at full size stay readable: from `OVERLAY_MIN` px they shrink to 3/4 at most, from `LEGIBLE` to `LEGIBLE`, then go over their own line (`row.over`: outlined, no backing). The e2e `carton` (stacked headlines) and `stack` (a video title's three tightly stacked lines) scenes pin the overlay case.
  - Fine print (under `LEGIBLE` at full size) shrinks to 1/4, then the side with the least `overlapArea` wins.
  - `paint()` fades each row in by its on-screen size (`g.size × scale.s`): hidden below `FADE_FROM` (4 px), full from `LEGIBLE` (6 px). A line whose row isn't fully shown gets the light yellow dashed outline, like live mode's skipped lines: zoom in. Outlines keep their on-screen width at any zoom. The e2e `page` scene (a dense paragraph) checks rows are hidden at 1× and all shown at 4× on the same layout.
  - Vertical lines get furigana-style labels to the right.
  - `layout()` returns a plan (box outlines, label groups with backing rects and text positions). `paint()` draws a plan through a transform, and `draw()` = both, for live mode.
  - `cjkStats.labels` (lines, labels, crowded, below, overlaps, hidden at this zoom, min/max size) is what the e2e checks read.
- **`reading.js`** must stay a classic IIFE script. kuromoji is UMD-only, so the reading worker is a classic worker that uses `importScripts`. The Node test `require()`s the same file. It reads `pinyinPro`, `wanakana` and `kuromoji` as globals and exposes `Reading.{zh,ja,ko,initJapanese}`. kuromoji and its 18 MB dictionary load only when Japanese is picked.
- **Position contract between the two halves:** each OCR line has `chars: [{c, x, y}]`, one entry per code point of `text`. Each reading unit `{i, n, parts, roman}` indexes code points of that same text, and `app.js` places a label at `line.chars.slice(u.i, u.i + u.n)`. So reading code must keep the code-point count unchanged. `fixDash` (ー→一) swaps one character for one character.
- **One read at a time:** every session run and creation in `ocr-worker.js` goes through `serial()`.
  - Messages keep arriving mid-read (a snap during a live read, a language switch). Interleaved runs break the WebGPU backend: "Failed to download data from buffer" on iOS, and a hang on SwiftShader.
  - A `cancel` message (sent by `readStill`) makes earlier reads stop at their next check, through `runOCR`'s `stop` option. They reject with `cancelled: true`, which `app.js` ignores.
  - If a WebGPU run fails anyway, `toWasm()` rebuilds every session on WASM from the cached model bytes and posts a `backend` event.
- **Backend:** `ocr-worker.js` creates each session with exactly one backend, and with `allow_spinning` off unless `?spin=1`. It reports which backend actually ran: WebGPU falls back to WASM with a `fallback` reason rather than silently.
- **Settings:** `ep`, `threads`, `spin` and `debug` come from the URL switch or, failing that, from `localStorage.dev`. That's written by the dev panel, opened with three taps on `#status`, because home-screen apps take no URL switches.
- **iOS status bar:** `apple-mobile-web-app-status-bar-style` must stay `black`. With `black-translucent`, iOS home-screen apps get a window one status bar shorter than the screen, pinned to the top. That leaves an unpaintable black strip at the bottom, which CSS can't fix.
- **Models:** the detector `det.*.onnx` (PP-OCRv6 tiny) is shared; `detBoxes` defaults in `ocr-core.js` and `_boxes` in `ocr_ref.py` hold its DB post-processing settings from its `inference.yml`. The recognizer comes from `MODEL` in `app.js`: zh and ja use `rec_zhja` (PP-OCRv6 small), ko uses `rec_ko` (PP-OCRv5 Korean; v6 has no Hangul). `prepare_models.py` reads the v6 dictionary from the model's `inference.yml` and converts the opset-11 v6 recognizer to opset 13 before quantizing. Each recognizer has a dictionary in which output index `i+1` is line `i`, index 0 is the CTC blank, and the last class is a space. "int8" means weight-only quantization with `DequantizeLinear`, and compute stays fp32, so the same files run on both WASM and WebGPU.
- **Service worker (`sw.js`):**
  - It adds the COOP/COEP headers that WASM threads (SharedArrayBuffer) need, since GitHub Pages can't send them. `app.js` reloads once to become cross-origin isolated, using the sessionStorage key `coi-reload`.
  - Two caches:
    - `SHELL`: app code, network-first with `no-cache`.
    - `ASSETS`: anything under `/models/` or `/vendor/`, cache-first. A miss is stored with `event.waitUntil` while it streams to the page. Awaiting `cache.put()` first would hold back every byte until the download finished, which breaks the progress bar.
  - **Bump `ASSETS` whenever model or vendor files change**, or installed apps keep the old ones.
  - A new top-level app file also needs adding to `SHELL_FILES`.
