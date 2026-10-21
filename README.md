# [CJK Reader](https://reinbowl.github.io/CJK-Reader/)

A PWA that points the phone camera at Chinese, Japanese or Korean text and shows how to read it:
pinyin for Chinese, furigana and romaji for Japanese, Revised Romanization for Korean. It reads
the text, it doesn't translate it. Everything runs on the device, and it works offline after the
first visit.

- **Live:** readings float above each line of text in the camera view. It reads when the camera
  is held steady on something new, and rests while nothing changes.
- **Shutter:** freezes the frame and re-reads it at higher resolution, with the readings drawn over
  it. The list button shows every line with ruby text.
- **Photo:** reads a picture from the library instead of the camera.

## How it works

```
camera frame ──► ocr-worker.js (module worker)
                   ocr-core.js: DB text detection → rotated crops → CTC recognition
                   onnxruntime-web, WASM (default) or WebGPU
             ──► reading-worker.js (classic worker)
                   reading.js: pinyin-pro (zh) · kuromoji + wanakana (ja) · RR rules (ko)
             ──► app.js draws each reading over its characters
```

Live mode doesn't read every frame:

- **Motion gate:** every 100 ms, `app.js` shrinks the view to a small grey thumbnail and compares it
  with the previous one, and with the one from the last read.
  - A fast move (or a new scene) fades the labels out, and nothing is read.
  - Slower movement and hand tremor move the labels along with the text instead.
  - A read starts once the view has been steady for about 0.2 s and has drifted about 30 px or
    changed since the last read.
  - What counts as "the same scene" adapts to how much the image jitters while steady. Big, sharp
    text close to the camera jitters more than a distant sign.
- **One at a time:** reads never overlap. The shutter cancels a live read that's still running, so
  the still read starts right away. If WebGPU fails mid-session (iOS can drop the GPU), the app
  switches to WASM until the next launch, and the status bar says so.
- **Pacing:** reads are at least 250 ms apart, and the gap is never shorter than the last read
  took, so the OCR worker is busy at most half the time.
- **Small lines:** lines thinner than 14 px on screen aren't recognised in live mode (the dev panel
  switches between 10, 14 and 20 px). They get a dashed outline; move closer or freeze to read them.

Labels:

- **Size:** each line gets one row of labels, sized from that line's own characters, so a big
  headline gets big pinyin and small print gets small pinyin. The row font then shrinks until the
  whole row fits the line (and the screen), but never below 8 px.
- **Position:** each label sits as close as it can to its own characters. Labels are only pushed
  sideways as far as needed to stop them overlapping (a least-squares fit).
- **Crowding:** the row goes above the line, or below if the space above is taken by other text or
  another row, or shrinks to fit the gap.
  - **Big text** shrinks to 3/4 of its size at most. Tightly stacked headlines leave no gap at all,
    so the row then goes over the line's own characters instead: outlined labels with no backing,
    so the big glyphs stay visible between the letters.
  - **Small text** shrinks to 8 px, then goes on the side that covers least.
  - Either way, every reading is always shown.
- **Freeze:** after the shutter, the live labels stay up while the frame is re-read at higher
  resolution, then every line gets its readings, small ones included.

Models: PaddleOCR's PP-OCRv6 (PP-OCRv5 for Korean, which v6 doesn't cover), in ONNX, with
weights stored as int8.

| file | model | size | used for |
|---|---|---|---|
| `det.int8.onnx` | PP-OCRv6 tiny detector | 0.6 MB | finding text lines (all languages) |
| `rec_zhja.int8.onnx` + `dict_zhja.txt` | PP-OCRv6 small recognizer | 5.6 MB | Simplified and Traditional Chinese, Japanese, English |
| `rec_ko.int8.onnx` + `dict_ko.txt` | PP-OCRv5 mobile Korean recognizer | 3.6 MB | Korean, English |

PP-OCRv6's own tiny recognizer is faster, but it has no kana and is missing common traditional
characters (請), so it can't serve Japanese or traditional Chinese. Its small detector is more
accurate than tiny but about 4× the size and slower.

"int8" here means **weight-only** quantization. Weights are stored as per-channel int8, and a
`DequantizeLinear` node expands them back to float32 at load/run time. Files are about 4× smaller
than fp32 (9.8 MB vs 36.3 MB), and the character error rate on the test images stays within 0.015
of fp32. Compute stays float32, so the same files run on the WASM and WebGPU backends.

Downloads are lazy and cached by the service worker:

- Core app: ~1 MB
- ONNX Runtime WASM: 14 MB
- Detector + current language's recognizer: 4–6 MB
- Japanese dictionary (kuromoji): 18 MB, only when 日本語 is picked
- ONNX Runtime WebGPU build: 27 MB, only when WebGPU is chosen and the device has a GPU adapter

## Layout

```
docs/                 ← the site (GitHub Pages serves this folder)
  index.html app.js style.css sw.js manifest.webmanifest icons/
  ocr-core.js ocr-worker.js reading.js reading-worker.js
  models/             ← generated by tools/prepare_models.py
  vendor/             ← copied from node_modules by tools/vendor.mjs
tools/
  prepare_models.py   download pinned models, verify sha256, int8-quantize → docs/models
  ocr_ref.py          reference pipeline in Python (PaddleOCR's own pre/post-processing), fp32 vs int8 eval
  make_test_images.py render test/images (signs + a calendar page) from Noto CJK fonts
  make_icons.py       render docs/icons
  vendor.mjs          copy onnxruntime-web, pinyin-pro, wanakana, kuromoji into docs/vendor
test/
  ocr.test.mjs        runs docs/ocr-core.js on test/images via onnxruntime-web, compares with Python
  reading.test.cjs    pinyin / romaji / Korean romanization cases
  e2e.test.mjs        Playwright + headless Chromium with a fake camera fed from test/images
```

## Build

```sh
npm ci                                  # JS deps (versions pinned)
npm run vendor                          # → docs/vendor
uv run tools/prepare_models.py          # → docs/models (add --fp32 to also keep float32 copies)
npm test                                # OCR pipeline + reading tests
npm run e2e                             # the app in headless Chromium (screenshots in .cache/e2e/)
npm run serve                           # http://localhost:8080 (localhost counts as secure for the camera)
```

The site needs HTTPS for the camera. `localhost` is fine, and on a phone use the GitHub Pages URL
(or a tunnel to your dev server).

## Deploy to GitHub Pages

Commit everything, including `docs/models` and `docs/vendor`. Then go to **Settings → Pages →
Deploy from a branch → `main` / `/docs`**.

- **File sizes:** the largest file is ~27 MB, well under GitHub's 100 MB limit.
- **Headers:** GitHub Pages can't send the COOP/COEP headers that multithreaded WASM needs, so
  `sw.js` adds them. On first visit the page reloads once to pick them up. The status bar shows
  `wasm×4` when threads are working.
- **Cache:** app code (HTML, JS, CSS) is checked with the server on every load, so a deploy
  shows up on the next reload. Models and vendor files are served cache-first: **bump `ASSETS` in
  `sw.js`** whenever you change them, otherwise installed apps keep the old files. When `sw.js`
  itself changes, open pages reload once to switch to it.

Installing on a phone:

- **iPhone:** Safari → Share → Add to Home Screen.
- **Android:** Chrome → Install app.

## Performance testing

The status bar shows detector and recognizer time per read, the number of lines (plus small ones
skipped), and the backend actually in use. URL switches:

| param | default | |
|---|---|---|
| `ep` | `wasm` | `webgpu` to try the WebGPU backend |
| `precision` | `int8` | `fp32` needs the fp32 files deployed (see `.gitignore`) |
| `threads` | `min(4, cores)` | `1` to compare single-threaded |
| `det` | `640` | detector input size in live mode (longest side); lower is faster, higher finds smaller text |
| `spin` | `0` | `1` lets ONNX Runtime's threads busy-wait between ops (its default; costs battery) |
| `debug` | off | `1` shows the motion gate's numbers and the viewport/safe-area sizes under the status bar |

A home-screen app can't be given URL switches. Instead, **tap the status line three times** to open
a dev panel with the debug readout and buttons for `ep`, `threads`, the live-mode minimum line size
and `spin`. The buttons reload
the app, and the choices are saved on the device. A URL switch overrides the panel.

Recognition time grows with the total width of the lines in view. Each crop is padded only to a
multiple of 32 px (64 px minimum), not to PaddleOCR's 320 px, so short boxes such as calendar
dates are cheap. The calendar page in `test/images` has 73 boxes:

| desktop, WASM, 960 px | detection | recognition |
|---|---|---|
| 1 thread | 290 ms | 1680 ms |
| 4 threads | | 660 ms (1920 ms with 320 px padding) |

On an iPhone, 6 lines took about 300–500 ms and a calendar about 3000 ms with PP-OCRv5 and 320 px
padding.

## Known limitations

- **Japanese readings** come from a word dictionary (IPADIC via kuromoji), so unusual names or
  readings can be wrong. Sound changes in counters aren't applied (一杯 shows "ichi hai", not
  "ippai").
- **Korean romanization** applies the main sound-change rules: liaison, nasal assimilation, ㄹ
  assimilation, aspiration, and palatalization. It doesn't do ㄴ-insertion in compound words, which
  needs a dictionary (서울역 shows "seouryeok"; official RR is "Seoullyeok").
- **Chinese readings:** pinyin-pro picks readings for characters with more than one from context,
  and applies the tone changes for 一 and 不 (yí gè, bú duì).
- **Picking a language:** you choose the language yourself. The Chinese/Japanese model doesn't
  read Hangul, and kanji-only text can't tell Chinese from Japanese.
- **ー vs 一:** the recognizer sometimes reads 一 (one) as ー (long-vowel mark). `reading.js` fixes
  that from the surrounding characters.
- **Crowded text:** when a line's labels can't go above or below it without covering other text,
  they're drawn at 8 px on the side that covers least, and can hide part of a neighbouring line.
  The list (after the shutter) always shows every line in full.

## Sources and licenses

- **PP-OCRv6 models:** PaddleOCR, Apache-2.0, from Hugging Face `PaddlePaddle/PP-OCRv6_tiny_det_onnx`
  and `PaddlePaddle/PP-OCRv6_small_rec_onnx`. **PP-OCRv5 Korean recognizer:** PaddleOCR, Apache-2.0,
  ONNX conversion from `monkt/paddleocr-onnx`. All pinned to commits in `tools/prepare_models.py`.
- **onnxruntime-web:** MIT.
- **pinyin-pro:** MIT.
- **wanakana:** MIT.
- **kuromoji.js:** Apache-2.0. Its dictionary is IPADIC; see `docs/vendor/kuromoji/NOTICE.md`.
