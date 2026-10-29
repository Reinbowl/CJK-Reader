# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["transformers==4.46.3", "sentencepiece==0.2.0", "sacremoses==0.1.1", "protobuf<6"]
# ///
"""Token ids from Hugging Face's MarianTokenizer for the translation test sentences, so the Node test can
check docs/marian.js tokenizes exactly the same way. Writes test/marian_ref.json.

    uv run tools/marian_ref.py
"""

import json
import re
from pathlib import Path

from transformers import MarianTokenizer

ROOT = Path(__file__).resolve().parent.parent
# Each pair's repo and pinned commit, from prepare_models.py (whose imports this environment lacks).
MARIAN = dict(re.findall(r'"([\w-]+)": f"\{HF\}/([^"]+)/resolve/', (ROOT / "tools/prepare_models.py").read_text()))
REV = dict(re.findall(r'"[\w-]+": f"\{HF\}/([^"]+)/resolve/([0-9a-f]{40})"', (ROOT / "tools/prepare_models.py").read_text()))

truth = json.loads((ROOT / "test/images/truth.json").read_text(encoding="utf-8"))
EXTRA = {
    "ja-en": ["つくえに はなが____。", "日本語を勉強するのは楽しいです。毎日少しずつ練習しています。", "ＡＢＣ１２３　全角", "😀 絵文字", "東京  タワー\t行き"],
    "ko-en": ["책상 위에 꽃이 있습니다.", "서울에서 부산까지 KTX로 2시간 30분 걸려요!", "😀 이모지"],
    "zh-en": ["学习一门新的语言需要时间，也需要耐心。", "請勿吸煙", "营业时间 9:00-17:00（周一休息）", "😀 表情"],
}
LANG = {"ja-en": "ja", "ko-en": "ko", "zh-en": "zh"}

out = {}
for pair, repo in MARIAN.items():
    tok = MarianTokenizer.from_pretrained(repo, revision=REV[repo])
    texts = [line for t in truth.values() if t["lang"] == LANG[pair] for line in t["lines"]] + EXTRA[pair]
    out[pair] = [{"text": t, "ids": tok(t)["input_ids"]} for t in dict.fromkeys(texts)]
    print(pair, len(out[pair]), "sentences")
(ROOT / "test/marian_ref.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
