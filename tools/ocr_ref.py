"""Reference PP-OCR pipeline in Python (mirrors PaddleOCR's own pre/post-processing).

Used to check the models and the int8 conversion, and as ground truth for the JS port.

    uv run tools/ocr_ref.py                 # evaluate fp32 vs int8 on test/images
    uv run tools/ocr_ref.py --json out.json # also dump int8 results for the JS comparison
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "docs" / "models"
IMAGES = ROOT / "test" / "images"


class OCR:
    def __init__(self, rec_model: str, dict_file: str, precision: str = "int8", det_limit: int = 960):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.det = ort.InferenceSession(str(MODELS / f"det.{precision}.onnx"), opts)
        self.rec = ort.InferenceSession(str(MODELS / f"{rec_model}.{precision}.onnx"), opts)
        self.chars = (MODELS / dict_file).read_text(encoding="utf-8").rstrip("\n").split("\n") + [" "]
        self.det_limit = det_limit

    # --- detection (DB) ---
    def detect(self, img):
        h, w = img.shape[:2]
        r = min(1.0, self.det_limit / max(h, w))
        rh, rw = max(32, round(h * r / 32) * 32), max(32, round(w * r / 32) * 32)
        x = cv2.resize(img, (rw, rh)).astype(np.float32) / 255.0
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        pred = self.det.run(None, {"x": x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        return self._boxes(pred, w / rw, h / rh)

    # DBPostProcess settings of the PP-OCRv6 tiny detector (its inference.yml).
    def _boxes(self, pred, sx, sy, thresh=0.2, box_thresh=0.4, unclip_ratio=1.4):
        bitmap = (pred > thresh).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours[:1000]:
            pts, sside = self._mini_box(c)
            if sside < 3:
                continue
            if self._score(pred, pts) < box_thresh:
                continue
            poly = pyclipper.PyclipperOffset()
            area, length = cv2.contourArea(pts), cv2.arcLength(pts, True)
            poly.AddPath(pts.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = poly.Execute(area * unclip_ratio / length)
            if len(expanded) != 1:
                continue
            pts, sside = self._mini_box(np.array(expanded[0]).reshape(-1, 1, 2))
            if sside < 5:
                continue
            pts = pts * [sx, sy]
            boxes.append(pts.astype(np.float32))
        return sorted(boxes, key=lambda b: (b[0][1], b[0][0]))

    @staticmethod
    def _mini_box(contour):
        rect = cv2.minAreaRect(contour)
        p = sorted(cv2.boxPoints(rect).tolist(), key=lambda q: q[0])
        left = sorted(p[:2], key=lambda q: q[1])
        right = sorted(p[2:], key=lambda q: q[1])
        return np.array([left[0], right[0], right[1], left[1]], np.float32), min(rect[1])

    @staticmethod
    def _score(pred, pts):
        h, w = pred.shape
        xmin, xmax = int(np.clip(np.floor(pts[:, 0].min()), 0, w - 1)), int(np.clip(np.ceil(pts[:, 0].max()), 0, w - 1))
        ymin, ymax = int(np.clip(np.floor(pts[:, 1].min()), 0, h - 1)), int(np.clip(np.ceil(pts[:, 1].max()), 0, h - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
        cv2.fillPoly(mask, [(pts - [xmin, ymin]).astype(np.int32)], 1)
        return cv2.mean(pred[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    # --- recognition (CTC) ---
    @staticmethod
    def crop(img, pts):
        w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
        h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
        M = cv2.getPerspectiveTransform(pts, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
        out = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
        return np.rot90(out) if h / max(w, 1) >= 1.5 else out

    # Like docs/ocr-core.js (not PaddleOCR): crops are padded to a 64 px minimum rather than 320,
    # widths round up to 32, and a batch holds as many ratio-sorted crops as fit 1280 columns.
    @staticmethod
    def rec_width(ratio, min_w=64):
        return max(min_w, math.ceil(48 * ratio / 32) * 32)

    def recognize(self, crops, cols=1280):
        ratios = [c.shape[1] / c.shape[0] for c in crops]
        order = np.argsort(ratios, kind="stable")
        results = [None] * len(crops)
        s = 0
        while s < len(order):
            e = s + 1
            while e < len(order) and (e - s + 1) * self.rec_width(ratios[order[e]]) <= cols:
                e += 1
            idx = order[s:e]
            s = e
            W = self.rec_width(ratios[idx[-1]])
            x = np.zeros((len(idx), 3, 48, W), np.float32)
            for j, i in enumerate(idx):
                c = crops[i]
                rw = min(W, int(math.ceil(48 * c.shape[1] / c.shape[0])))
                r = cv2.resize(c, (rw, 48)).astype(np.float32) / 255.0
                x[j, :, :, :rw] = ((r - 0.5) / 0.5).transpose(2, 0, 1)
            probs = self.rec.run(None, {"x": x})[0]
            for j, i in enumerate(idx):
                results[i] = self._ctc(probs[j])
        return results

    def _ctc(self, p):
        ids, conf = p.argmax(1), p.max(1)
        text, scores, prev = [], [], 0
        for t, k in enumerate(ids):
            if k != 0 and k != prev:
                text.append(self.chars[k - 1])
                scores.append(conf[t])
            prev = k
        return "".join(text), float(np.mean(scores)) if scores else 0.0

    def __call__(self, img):
        boxes = self.detect(img)
        recs = self.recognize([self.crop(img, b) for b in boxes]) if boxes else []
        return [(b, t, s) for b, (t, s) in zip(boxes, recs) if s >= 0.5]


def cer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return d[-1] / max(1, len(ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    truth = json.loads((IMAGES / "truth.json").read_text(encoding="utf-8"))
    dump = {}
    for precision in ("fp32", "int8"):
        engines = {"zhja": OCR("rec_zhja", "dict_zhja.txt", precision), "ko": OCR("rec_ko", "dict_ko.txt", precision)}
        print(f"== {precision}")
        for name, t in truth.items():
            img = cv2.imread(str(IMAGES / f"{name}.png"))
            eng = engines["ko" if t["lang"] == "ko" else "zhja"]
            t0 = time.perf_counter()
            res = eng(img)
            ms = (time.perf_counter() - t0) * 1000
            got = [r[1] for r in res]
            errs = [min((cer(line, g) for g in got), default=1.0) for line in t["lines"]]
            print(f"  {name:8s} {len(res)} lines  CER {np.mean(errs):.3f}  {ms:.0f} ms  {got}")
            if precision == "int8":
                dump[name] = [{"box": b.round(1).tolist(), "text": tx, "score": round(s, 4)} for b, tx, s in res]
    if args.json:
        args.json.write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
