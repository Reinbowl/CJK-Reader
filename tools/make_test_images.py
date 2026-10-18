"""Render synthetic sign/label photos with known text for checking the OCR pipeline.

    uv run tools/make_test_images.py --fonts /path/to/noto-cjk-otfs

Writes test/images/*.png and test/images/truth.json.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test" / "images"

SCENES = {
    "zh": ("NotoSansCJKsc-Regular.otf", ["欢迎光临北京大学", "出口在左边请注意安全", "今天天气很好，我们去公园散步。", "银行营业时间 9:00-17:00", "重庆火锅", "长城"]),
    "zh_trad": ("NotoSansCJKtc-Regular.otf", ["歡迎來到台灣", "請勿吸煙", "捷運站出口"]),
    "ja": ("NotoSansCJKjp-Regular.otf", ["東京駅はこちらです", "ラーメン一杯八百円", "日本語の勉強は楽しい", "出口 改札口", "今日は雨が降っています"]),
    "ko": ("NotoSansCJKkr-Regular.otf", ["서울역 출구는 왼쪽입니다", "안녕하세요 반갑습니다", "한국어를 공부합니다", "영업시간 오전 10시", "신라호텔", "같이 먹어요"]),
}


def render(lines, font_path, rng, size=(1280, 720)):
    w, h = size
    base = np.full((h, w, 3), rng.integers(200, 245, 3), np.uint8)
    base = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    y = 40
    for i, text in enumerate(lines):
        px = int(rng.integers(32, 64))
        font = ImageFont.truetype(str(font_path), px)
        box = font.getbbox(text)
        tw, th = box[2] - box[0] + 20, box[3] - box[1] + 20
        inverted = i % 4 == 3  # some light-on-dark signs
        layer = Image.new("RGBA", (tw, th), (30, 60, 140, 255) if inverted else (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((10 - box[0], 10 - box[1]), text, font=font,
                                   fill=(250, 250, 250, 255) if inverted else (20, 20, 20, 255))
        angle = float(rng.uniform(-8, 8)) if i % 3 == 1 else 0.0
        layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
        x = int(rng.integers(30, max(31, w - layer.width - 30)))
        if y + layer.height > h - 10:
            break
        img.paste(layer, (x, y), layer)
        y += layer.height + int(rng.integers(15, 40))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", required=True, type=Path)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    random.seed(7)
    truth = {}
    for name, (font, lines) in SCENES.items():
        img = render(lines, args.fonts / font, rng)
        img.save(OUT / f"{name}.png")
        truth[name] = {"lang": name.split("_")[0], "lines": lines}
    (OUT / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(truth)} images to {OUT}")


if __name__ == "__main__":
    main()
