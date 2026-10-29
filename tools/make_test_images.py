"""Render synthetic sign/label photos with known text for checking the OCR pipeline.

    uv run tools/make_test_images.py --fonts /path/to/noto-cjk-otfs
    uv run tools/make_test_images.py --fonts .cache/fonts --scenes calendar carton page menu subs clock   # just these

Writes test/images/*.png and test/images/truth.json (entries for scenes not rendered are kept).
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


# A wall calendar page for October 2026: lots of short, small boxes (dates, lunar days, festivals),
# the case where recognition cost blows up. Lunar dates are plausible, not authoritative.
LUNAR = ["初一", "初二", "初三", "初四", "初五", "初六", "初七", "初八", "初九", "初十",
         "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十",
         "廿一", "廿二", "廿三", "廿四", "廿五", "廿六", "廿七", "廿八", "廿九", "三十"]
FESTIVALS = {1: "国庆节", 8: "寒露", 10: "九月", 18: "重阳节", 23: "霜降"}


def calendar_lines():
    # 1 Oct 2026 is 廿一 of the 8th lunar month (29 days); the 9th month starts on 10 Oct.
    cells = []
    for day in range(1, 32):
        lunar = LUNAR[20 + day - 1] if day < 10 else LUNAR[day - 10]
        cells.append((str(day), FESTIVALS.get(day, lunar)))
    return cells


def render_calendar(font_path, rng, size=(960, 1280)):
    w, h = size
    base = np.full((h, w, 3), (246, 243, 236), np.uint8)
    base = np.clip(base + rng.normal(0, 5, base.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    d = ImageDraw.Draw(img)
    font = lambda px: ImageFont.truetype(str(font_path), px)
    red, ink, grey = (196, 30, 36), (30, 30, 30), (110, 110, 110)
    lines = []

    def text(xy, s, px, fill, anchor="mm"):
        d.text(xy, s, font=font(px), fill=fill, anchor=anchor)
        lines.append(s)

    text((w // 2, 90), "2026年10月", 68, red)
    text((w // 2, 170), "丙午马年 农历八月", 34, grey)
    x0, cw, y0 = 30, (w - 60) / 7, 250
    for c, s in enumerate("日一二三四五六"):
        text((x0 + cw * (c + 0.5), y0), s, 38, red if c in (0, 6) else ink)
    rh, top = 150, y0 + 50
    for r in range(6):
        d.line([(x0, top + r * rh), (w - x0, top + r * rh)], fill=(215, 210, 200), width=2)
    for i, (num, sub) in enumerate(calendar_lines()):
        col, row = (4 + i) % 7, (4 + i) // 7
        cx, cy = x0 + cw * (col + 0.5), top + row * rh + rh / 2
        colour = red if col in (0, 6) or sub in FESTIVALS.values() else ink
        text((cx, cy - 22), num, 50, colour)
        text((cx, cy + 34), sub, 24, red if sub in FESTIVALS.values() else grey)
    text((w * 0.32, h - 70), "宜 出行 祭祀", 34, grey)
    text((w * 0.72, h - 70), "忌 动土", 34, grey)
    return img.filter(ImageFilter.GaussianBlur(0.6)), lines


# Big headlines stacked with almost no gap, like a milk carton: no room above or below for labels.
CARTON = [("特選", 100), ("北海道牛乳", 120), ("種類別牛乳", 72), ("常温保存可能品", 85)]


def render_carton(font_path, rng, size=(720, 640)):
    w, h = size
    base = np.full((h, w, 3), (244, 246, 248), np.uint8)
    img = Image.fromarray(np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    y = 70
    for text, px in CARTON:
        d.text((w // 2, y + px // 2), text, font=ImageFont.truetype(str(font_path), px), fill=(25, 70, 160), anchor="mm")
        y += px + 12
    return img.filter(ImageFilter.GaussianBlur(0.6)), [t for t, _ in CARTON]


# A book page: a heading over a paragraph of small text with tight leading, the case where readings
# have only a thin gap between lines. Lines are wrapped by width, with a two-character indent.
PAGE_TITLE = "读书的乐趣"
PAGE_TEXT = ("学习一门新的语言需要时间，也需要耐心。刚开始的时候，很多字都不认识，读一句话要查好几次字典。"
             "但是只要每天坚持读一点，慢慢就会发现，自己能看懂的东西越来越多。书店里的小说、街上的招牌、"
             "朋友发来的消息，都变成了练习的材料。遇到不会读的字，可以用手机拍下来，看看它的拼音，再大声地"
             "念几遍。这样的方法虽然简单，却非常有效。语言不只是一种工具，更是一扇通往另一种文化的窗户。"
             "每读懂一句话，你就离那个世界更近了一步。")


def render_page(font_path, rng, size=(1100, 560)):
    w, h = size
    base = np.full((h, w, 3), (238, 236, 230), np.uint8)
    img = Image.fromarray(np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    ink = (35, 35, 35)
    d.text((w // 2, 90), PAGE_TITLE, font=ImageFont.truetype(str(font_path), 60), fill=ink, anchor="mm")
    font, px, pitch, left, right = ImageFont.truetype(str(font_path), 24), 24, 36, 70, w - 70
    lines, line, x0 = [], "", left + 2 * px
    for ch in PAGE_TEXT:
        if x0 + font.getlength(line + ch) > right:
            lines.append((x0, line))
            line, x0 = "", left
        line += ch
    lines.append((x0, line))
    for i, (x, text) in enumerate(lines):
        d.text((x, 180 + i * pitch), text, font=font, fill=ink)
    return img.filter(ImageFilter.GaussianBlur(0.6)), [PAGE_TITLE] + [t for _, t in lines]


# A menu: dishes stacked with tight leading and a price at the far right of each, so the detector
# reads names and prices as separate lines. Each dish is its own paragraph, not one run-on name.
MENU_TITLE = "凉菜篇"
MENU = [("川北凉粉", 8), ("酸辣蕨根粉", 10), ("干炸小黄鱼", 12), ("凉拌苦瓜", 10), ("拌金针菇", 10), ("口水鸡", 20),
        ("夫妻肺片", 20), ("凉拌海带丝", 8), ("小葱拌豆腐", 8), ("红油耳丝", 20)]


def render_menu(font_path, rng, size=(640, 760)):
    w, h = size
    base = np.full((h, w, 3), (240, 170, 60), np.uint8)
    img = Image.fromarray(np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.text((w // 2, 70), MENU_TITLE, font=ImageFont.truetype(str(font_path), 64), fill=(200, 30, 30), anchor="mm")
    font, ink = ImageFont.truetype(str(font_path), 40), (25, 20, 15)
    for i, (dish, price) in enumerate(MENU):
        y = 140 + i * 58
        d.text((70, y), dish, font=font, fill=ink)
        d.text((w - 70, y), f"{price}元", font=font, fill=ink, anchor="ra")
    return img.filter(ImageFilter.GaussianBlur(0.6)), [MENU_TITLE] + [s for dish, price in MENU for s in (dish, f"{price}元")]


# A video frame: a title of three lines stacked with almost no gap, and a subtitle at the bottom.
# subs_a and subs_b differ only in the subtitle, like two moments of the same video (subs_gap has
# none); clock_a and clock_b only in a clock in the middle of the picture, away from the Chinese.
SUBS_TITLE = ["今晚的月亮又圆又亮", "我们一起去海边走走吧", "听听风和海浪的声音"]
SUBS = {"subs_a": "你今天过得怎么样", "subs_b": "明天我们再见面吧", "subs_gap": ""}
CLOCKS = {"clock_a": "12:41", "clock_b": "12:42 PM"}


def render_subs(font_path, subtitle, size=(1280, 720), clock=None):
    w, h = size
    rng = np.random.default_rng(19)  # same background for every subtitle
    ramp = np.linspace(0, 1, h)[:, None, None]
    base = (1 - ramp) * np.array([40, 36, 90]) + ramp * np.array([18, 16, 40])
    img = Image.fromarray(np.clip(np.broadcast_to(base, (h, w, 3)) + rng.normal(0, 5, (h, w, 3)), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for i, text in enumerate(SUBS_TITLE):
        d.text((60, 60 + i * 72), text, font=ImageFont.truetype(str(font_path), 60), fill=(240, 240, 250))
    if subtitle:
        d.text((w // 2, 640), subtitle, font=ImageFont.truetype(str(font_path), 48), fill=(255, 255, 255), anchor="mm",
               stroke_width=2, stroke_fill=(0, 0, 0))
    if clock:
        d.text((1180, 450), clock, font=ImageFont.truetype(str(font_path), 56), fill=(250, 250, 250), anchor="rm")
    return img.filter(ImageFilter.GaussianBlur(0.6)), SUBS_TITLE + ([subtitle] if subtitle else []) + ([clock] if clock else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", required=True, type=Path)
    ap.add_argument("--scenes", nargs="*", help="only render these (default: all)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    want = set(args.scenes or [*SCENES, "calendar", "carton", "page", "menu", "subs", "clock"])
    rng = np.random.default_rng(7)
    random.seed(7)
    truth_path = OUT / "truth.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8")) if truth_path.exists() else {}
    # The sign scenes share one rng, so render all of them (keeping it in step) if any is wanted.
    for name, (font, lines) in SCENES.items() if want & set(SCENES) else ():
        img = render(lines, args.fonts / font, rng)
        if name in want:
            img.save(OUT / f"{name}.png")
            truth[name] = {"lang": name.split("_")[0], "lines": lines}
    if "calendar" in want:
        img, lines = render_calendar(args.fonts / "NotoSansCJKsc-Regular.otf", np.random.default_rng(11))
        img.save(OUT / "calendar.png")
        truth["calendar"] = {"lang": "zh", "lines": lines}
    if "carton" in want:
        # The Simplified Chinese font has the kanji and kana; glyph shapes differ a little from Japanese.
        img, lines = render_carton(args.fonts / "NotoSansCJKsc-Regular.otf", np.random.default_rng(13))
        img.save(OUT / "carton.png")
        truth["carton"] = {"lang": "ja", "lines": lines}
    if "page" in want:
        img, lines = render_page(args.fonts / "NotoSansCJKsc-Regular.otf", np.random.default_rng(17))
        img.save(OUT / "page.png")
        truth["page"] = {"lang": "zh", "lines": lines}
    if "menu" in want:
        img, lines = render_menu(args.fonts / "NotoSansCJKsc-Regular.otf", np.random.default_rng(23))
        img.save(OUT / "menu.png")
        truth["menu"] = {"lang": "zh", "lines": lines}
    if "subs" in want:
        for name, subtitle in SUBS.items():
            img, lines = render_subs(args.fonts / "NotoSansCJKsc-Regular.otf", subtitle)
            img.save(OUT / f"{name}.png")
            truth[name] = {"lang": "zh", "lines": lines}
    if "clock" in want:
        for name, clock in CLOCKS.items():
            img, lines = render_subs(args.fonts / "NotoSansCJKsc-Regular.otf", SUBS["subs_a"], clock=clock)
            img.save(OUT / f"{name}.png")
            truth[name] = {"lang": "zh", "lines": lines}
    truth_path.write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(want)} images to {OUT}")


if __name__ == "__main__":
    main()
