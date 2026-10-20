"""Download the OCR models (ONNX) and write int8 weight-only versions to docs/models.

  det       PP-OCRv6 tiny detector, for every language
  rec_zhja  PP-OCRv6 small recogniser: Chinese (simplified and traditional), Japanese, English
  rec_ko    PP-OCRv5 mobile Korean recogniser (PP-OCRv6 has no Korean)

Weight-only int8: every Conv/ConvTranspose/MatMul weight is stored as per-channel symmetric
int8 plus a float scale, and a DequantizeLinear node turns it back into float32 at runtime.
Files shrink ~4x, compute stays float32, so it runs unchanged on the WASM and WebGPU backends.

    uv run tools/prepare_models.py            # int8 (default)
    uv run tools/prepare_models.py --fp32     # also keep the original float32 files
"""

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper, numpy_helper, version_converter

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", action="store_true", help="also write the float32 models (*.fp32.onnx)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        src = fetch(name)
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


if __name__ == "__main__":
    main()
