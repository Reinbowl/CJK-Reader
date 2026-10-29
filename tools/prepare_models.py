"""Download the OCR models (ONNX) and write int8 weight-only versions to docs/models.

  det       PP-OCRv6 tiny detector, for every language
  rec_zhja  PP-OCRv6 small recogniser: Chinese (simplified and traditional), Japanese, English
  rec_ko    PP-OCRv5 mobile Korean recogniser (PP-OCRv6 has no Korean)

Translation into English, in docs/models/translate/<pair>/: Marian models exported to ONNX,
encoder.int8.onnx and decoder.int8.onnx (dynamic int8: MatMuls and embeddings), and tokenizer.json
(SentencePiece pieces and vocabulary, for docs/marian.js).
  zh-en  OPUS-MT
  ja-en  FuguMT
  ko-en  OPUS-MT

Weight-only int8: every Conv/ConvTranspose/MatMul weight is stored as per-channel symmetric
int8 plus a float scale, and a DequantizeLinear node turns it back into float32 at runtime.
Files shrink ~4x, compute stays float32, so it runs unchanged on the WASM and WebGPU backends.

    uv run tools/prepare_models.py            # int8 (default)
    uv run tools/prepare_models.py --fp32     # also keep the original float32 files
"""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper, numpy_helper, version_converter
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "models"
CACHE = ROOT / ".cache" / "models"

HF = "https://huggingface.co"
# Pinned to exact repo commits + sha256 so a changed upstream file fails loudly.
SOURCES = {
    "det.onnx": (
        f"{HF}/PaddlePaddle/PP-OCRv6_tiny_det_onnx/resolve/2ba1506c0380b8f0b03dd142459aac66d4421f6c/inference.onnx",
        "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8",
    ),
    "rec_zhja.onnx": (
        f"{HF}/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/b8f84f0b80c529de40b4fbb3544b84fa7233a513/inference.onnx",
        "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
    ),
    # PaddleOCR 3.x ships the character list inside inference.yml (PostProcess.character_dict).
    "dict_zhja.txt": (
        f"{HF}/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/b8f84f0b80c529de40b4fbb3544b84fa7233a513/inference.yml",
        None,
    ),
    "rec_ko.onnx": (
        f"{HF}/monkt/paddleocr-onnx/resolve/7b02d0a30a07ba2b92ad1ff5a8941ae2c633de65/languages/korean/rec.onnx",
        None,
    ),
    "dict_ko.txt": (
        f"{HF}/monkt/paddleocr-onnx/resolve/7b02d0a30a07ba2b92ad1ff5a8941ae2c633de65/languages/korean/dict.txt",
        None,
    ),
}

# Translators that run on onnxruntime-web like the OCR: Marian models (OPUS-MT for Chinese and Korean,
# FuguMT for Japanese) exported to ONNX, an encoder and a decoder without a KV
# cache. Their weights are int8-quantized here (dynamic quantization: int8 MatMuls and embeddings), and
# their SentencePiece model and vocabulary become one tokenizer.json for docs/marian.js.
MARIAN = {
    "ja-en": f"{HF}/Kadonox/fugumt-ja-en-onnx/resolve/9b09cd7e6263223ac4a5988da1d881ef02b55be8",
    "ko-en": f"{HF}/Xenova/opus-mt-ko-en/resolve/d9fa1ac6008242100fecb4fff9b0a5917a1be81f",
    "zh-en": f"{HF}/Xenova/opus-mt-zh-en/resolve/39d480d52a9ea3065a1f117adfe4dbc55de10e6f",
}
MARIAN_FILES = ["onnx/encoder_model.onnx", "onnx/decoder_model.onnx", "source.spm", "vocab.json", "config.json", "tokenizer_config.json"]
MARIAN_SHA = {
    "ja-en/onnx/encoder_model.onnx": "ce16e5f7cf1fde5dffbae8ed5a9c8d915de8084edc34ce5549a8e7c7271a5d2e",
    "ja-en/onnx/decoder_model.onnx": "f3fef884dc2bcaba1e3a6f844261ad31be84b433487156d2c11c0e8b297c15ae",
    "ja-en/source.spm": "2b75a55d81ba5be08d5cc971eecbb03d8efaf66382f2a4f9f0a5fe2fca2b6741",
    "ja-en/vocab.json": "bd868a2a41f8e2fd818fd575bb3f8fa5f1986859017de2badd2aa13971ef8769",
    "ja-en/config.json": "24e201e35f20db82adcd87924d59398bf986560a46532d01fb2ae19cf606f17f",
    "ja-en/tokenizer_config.json": "414e9e7e2492b95fa80f24ccb295645d35c4dd0a74d7ef8b89820c0c7e8ec356",
    "ko-en/onnx/encoder_model.onnx": "9c0b3965f9030e4d479d6d9f113cfdc2e61b074240cbc80c47895d94e684dd7c",
    "ko-en/onnx/decoder_model.onnx": "d07d3c7daa90c0d425b8f4a53fe8f221a9b071b2dac5d76a0c4eae52f7a018ee",
    "ko-en/source.spm": "c9496f7c2be9aecb84c751ae9f35a875915dde8e3892f652a5c76811ab2a0f49",
    "ko-en/vocab.json": "97197044d4803c664a37f8ca2ced9c3405b8a3df6b262a12da22d990b0d756cd",
    "ko-en/config.json": "8260080db48141328ae198ede2b25483fa4a6f46dc2859784d6e21c188865e43",
    "ko-en/tokenizer_config.json": "3df327b013e1fd1a7c7423a3c521d325cb0e641044f5ced6d2ec44ab811b4f0e",
    "zh-en/onnx/encoder_model.onnx": "555761f7077f36ee3e8349a8892002ae9995fa572381ebc6dbca322f99f6cd72",
    "zh-en/onnx/decoder_model.onnx": "fd368f87cc14f1c12c0ea1fc550f8125cd65b8a4649b628ecc76d05140a83405",
    "zh-en/source.spm": "e27a3a1b539f4959ec72ea60e453f49156289f95d4e6000b29332efc45616203",
    "zh-en/vocab.json": "08a119a1defd522fa047cb5e3bfe3e89633e96caa38ced0dc9cee7ef1021a011",
    "zh-en/config.json": "293d318fce41dbf04114eac45037bb88a32d7c4ee21011a75e24a8b98ca45ad1",
    "zh-en/tokenizer_config.json": "08849acc0a539c4749d8665e9d6217735503a97871ccebeea8a762d5fba1acf7",
}
# Credits, written next to each pair's files.
MARIAN_NOTICE = {
    "ja-en": "FuguMT ja-en by Satoshi Takahashi (staka), https://huggingface.co/staka/fugumt-ja-en, CC BY-SA 4.0.\n"
             "ONNX export: https://huggingface.co/Kadonox/fugumt-ja-en-onnx (CC BY-SA 4.0). int8-quantized by tools/prepare_models.py.\n",
    "ko-en": "OPUS-MT ko-en by the Language Technology Research Group at the University of Helsinki,\n"
             "https://huggingface.co/Helsinki-NLP/opus-mt-ko-en, Apache-2.0.\n"
             "ONNX export: https://huggingface.co/Xenova/opus-mt-ko-en. int8-quantized by tools/prepare_models.py.\n",
    "zh-en": "OPUS-MT zh-en by the Language Technology Research Group at the University of Helsinki,\n"
             "https://huggingface.co/Helsinki-NLP/opus-mt-zh-en, CC BY 4.0.\n"
             "ONNX export: https://huggingface.co/Xenova/opus-mt-zh-en. int8-quantized by tools/prepare_models.py.\n",
}
SOURCES.update({f"marian/{pair}/{f}": (f"{url}/{f}", MARIAN_SHA.get(f"{pair}/{f}")) for pair, url in MARIAN.items() for f in MARIAN_FILES})

# Which weight axis holds the output channels, per op type.
WEIGHT_AXIS = {"Conv": 0, "ConvTranspose": 1, "MatMul": 1}
MIN_WEIGHTS = 256  # leave tiny tensors in float


def fetch(name: str) -> Path:
    url, sha = SOURCES[name]
    path = CACHE / f"{hashlib.sha256(url.encode()).hexdigest()[:12]}-{url.rsplit('/', 1)[1]}"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name}")
        with urllib.request.urlopen(url) as r, open(path.with_suffix(".part"), "wb") as f:
            shutil.copyfileobj(r, f)
        path.with_suffix(".part").rename(path)
    if sha:
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(f"{name}: sha256 mismatch\n  expected {sha}\n  got      {got}")
    return path


def constants_to_initializers(g: onnx.GraphProto) -> None:
    # paddle2onnx stores the weights as Constant nodes; initializers are easier to rewrite.
    for node in list(g.node):
        if node.op_type == "Constant" and len(node.attribute) == 1 and node.attribute[0].name == "value":
            t = numpy_helper.to_array(node.attribute[0].t)
            g.initializer.append(numpy_helper.from_array(t, node.output[0]))
            g.node.remove(node)


def quantize_weights_int8(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    constants_to_initializers(g)
    inits = {t.name: t for t in g.initializer}
    done: dict[str, str] = {}
    new_nodes = []

    for node in g.node:
        axis = WEIGHT_AXIS.get(node.op_type)
        if axis is None or len(node.input) < 2:
            continue
        wname = node.input[1]
        if wname in done:
            node.input[1] = done[wname]
            continue
        t = inits.get(wname)
        if t is None or t.data_type != TensorProto.FLOAT:
            continue
        w = numpy_helper.to_array(t)
        if w.size < MIN_WEIGHTS or w.ndim < 2:
            continue

        reduce = tuple(i for i in range(w.ndim) if i != axis)
        scale = np.abs(w).max(axis=reduce) / 127.0
        scale[scale == 0] = 1.0
        shape = [1] * w.ndim
        shape[axis] = -1
        q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)

        g.initializer.remove(t)
        g.initializer.extend([
            numpy_helper.from_array(q, wname + "_q"),
            numpy_helper.from_array(scale.astype(np.float32), wname + "_scale"),
            numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), wname + "_zp"),
        ])
        out = wname + "_dq"
        new_nodes.append(helper.make_node(
            "DequantizeLinear", [wname + "_q", wname + "_scale", wname + "_zp"], [out],
            name=wname + "_DequantizeLinear", axis=axis,
        ))
        node.input[1] = out
        done[wname] = out

    # DequantizeLinear nodes only read initializers, so putting them first keeps the graph sorted.
    for n in reversed(new_nodes):
        g.node.insert(0, n)
    onnx.checker.check_model(model)
    print(f"  quantized {len(done)} weight tensors")
    return model


def untie(path: Path) -> onnx.ModelProto:
    """The decoder's output layer reads the embedding through a Transpose node, which the quantizer
    can't see through, so it would stay float32. Give it its own transposed copy: both become int8."""
    model = onnx.load(path)
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    for node in list(g.node):
        if node.op_type == "Transpose" and node.input[0] in inits:
            w = numpy_helper.to_array(inits[node.input[0]])
            perm = next((list(a.ints) for a in node.attribute if a.name == "perm"), None)
            g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(w.transpose(perm)), node.output[0]))
            g.node.remove(node)
    return model


def protobuf(buf: bytes) -> list[tuple[int, int | bytes]]:
    """The (field, value) pairs of one protobuf message: varints as ints, the rest as bytes."""
    out, i = [], 0
    def varint():
        nonlocal i
        v = shift = 0
        while True:
            b = buf[i]; i += 1
            v |= (b & 0x7F) << shift; shift += 7
            if b < 0x80: return v
    while i < len(buf):
        key = varint()
        field, wire = key >> 3, key & 7
        if wire == 0: out.append((field, varint()))
        elif wire == 1: out.append((field, buf[i:i + 8])); i += 8
        elif wire == 5: out.append((field, buf[i:i + 4])); i += 4
        elif wire == 2:
            n = varint(); out.append((field, buf[i:i + n])); i += n
        else: raise ValueError(f"protobuf wire type {wire}")
    return out


def marian_tokenizer(spm: Path, vocab: Path, config: Path) -> dict:
    """tokenizer.json for docs/marian.js: the source SentencePiece unigram pieces with their scores,
    the shared vocabulary (id -> token) and the special ids."""
    pieces, normalizer = [], {}
    for field, v in protobuf(spm.read_bytes()):
        if field == 1:  # ModelProto.pieces: piece = 1, score = 2, type = 3 (1 normal, 2 unknown, 3 control, 4 user-defined)
            f = dict(protobuf(v))
            if f.get(3, 1) in (1, 4):
                pieces.append([f[1].decode(), round(float(np.frombuffer(f.get(2, b"\0\0\0\0"), "<f4")[0]), 4)])
        elif field == 3:  # ModelProto.normalizer_spec: name = 1, add_dummy_prefix = 3, remove_extra_whitespaces = 4
            normalizer = {k: v for k, v in protobuf(v) if k in (1, 3, 4)}
    if normalizer.get(1, b"nmt_nfkc") not in (b"nmt_nfkc", b"nfkc") or normalizer.get(3, 1) != 1 or normalizer.get(4, 1) != 1:
        raise SystemExit(f"{spm}: unexpected SentencePiece normalizer {normalizer}")
    ids = json.loads(vocab.read_text(encoding="utf-8"))
    tokens = [None] * (max(ids.values()) + 1)
    for t, i in ids.items():
        tokens[i] = t
    c = json.loads(config.read_text(encoding="utf-8"))
    return {"pieces": pieces, "vocab": tokens, "unk": ids["<unk>"], "eos": c["eos_token_id"], "pad": c["pad_token_id"],
            "start": c["decoder_start_token_id"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", action="store_true", help="also write the float32 models (*.fp32.onnx)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        src = fetch(name)
        if name.startswith("marian/"):
            continue
        if name.endswith(".txt"):
            # Normalise to LF and drop a trailing blank line; index i+1 in the model output is line i.
            text = src.read_text(encoding="utf-8")
            if src.suffix == ".yml":
                lines = [str(c) for c in yaml.safe_load(text)["PostProcess"]["character_dict"]]
            else:
                lines = text.replace("\r\n", "\n").rstrip("\n").split("\n")
            (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"{name}: {len(lines)} characters")
            continue
        stem = name.removesuffix(".onnx")
        if args.fp32:
            shutil.copyfile(src, OUT / f"{stem}.fp32.onnx")
        print(f"{name}:")
        model = onnx.load(src)
        # Per-channel DequantizeLinear needs opset 13; the PP-OCRv6 recogniser is exported at 11.
        if next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")) < 13:
            model = version_converter.convert_version(model, 13)
        q = quantize_weights_int8(model)
        dst = OUT / f"{stem}.int8.onnx"
        onnx.save(q, dst)
        print(f"  {src.stat().st_size / 1e6:.1f} MB -> {dst.stat().st_size / 1e6:.1f} MB")

    for pair in MARIAN:
        out = OUT / "translate" / pair
        out.mkdir(parents=True, exist_ok=True)
        src = {f: fetch(f"marian/{pair}/{f}") for f in MARIAN_FILES}
        print(f"translate/{pair}:")
        for part in ["encoder", "decoder"]:
            dst = out / f"{part}.int8.onnx"
            quantize_dynamic(untie(src[f"onnx/{part}_model.onnx"]), dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])
            print(f"  {part}: {src[f'onnx/{part}_model.onnx'].stat().st_size / 1e6:.1f} MB -> {dst.stat().st_size / 1e6:.1f} MB")
        tok = marian_tokenizer(src["source.spm"], src["vocab.json"], src["config.json"])
        (out / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"  tokenizer: {len(tok['pieces'])} pieces, {len(tok['vocab'])} tokens")
        (out / "NOTICE").write_text(MARIAN_NOTICE[pair], encoding="utf-8")


if __name__ == "__main__":
    main()
