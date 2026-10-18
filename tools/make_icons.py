"""Render the app icons (読 with "dú" above it).

    uv run tools/make_icons.py --font /path/to/NotoSansCJKjp-Regular.otf
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "docs" / "icons"
BG, FG, ACCENT = (17, 17, 19), (244, 244, 245), (255, 224, 102)


def icon(size: int, font: Path, safe: float) -> Image.Image:
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)
    big = ImageFont.truetype(str(font), int(size * 0.5 * safe))
    small = ImageFont.truetype(str(font), int(size * 0.16 * safe))
    cx = size / 2
    d.text((cx, size * (0.5 + 0.08 * safe)), "読", font=big, fill=FG, anchor="mm")
    d.text((cx, size * (0.5 - 0.3 * safe)), "dú", font=small, fill=ACCENT, anchor="mm")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", required=True, type=Path)
    font = ap.parse_args().font
    OUT.mkdir(parents=True, exist_ok=True)
    icon(192, font, 1.0).save(OUT / "icon-192.png")
    icon(512, font, 1.0).save(OUT / "icon-512.png")
    icon(512, font, 0.8).save(OUT / "icon-maskable-512.png")  # content inside the maskable safe zone
    icon(180, font, 1.0).save(OUT / "apple-touch-icon.png")
    print(f"icons written to {OUT}")


if __name__ == "__main__":
    main()
