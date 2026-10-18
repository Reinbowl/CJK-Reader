"""Download the PP-OCRv5 mobile ONNX models and write int8 weight-only versions to docs/models.

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
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "models"
CACHE = ROOT / ".cache" / "models"

HF = "https://huggingface.co"
# Pinned to exact repo commits + sha256 so a changed upstream file fails loudly.
SOURCES = {
    "det.onnx": (
        f"{HF}/webnn/PP-OCRv5-ONNX/resolve/006b4419a415072fbb16a1d7d0d8548f789a8fae/ch_PP-OCRv5_det.onnx",
        "d7fe3ea74652890722c0f4d02458b7261d9f5ae6c92904d05707c9eb155c7924",
    ),
    "rec_zhja.onnx": (
        f"{HF}/webnn/PP-OCRv5-ONNX/resolve/006b4419a415072fbb16a1d7d0d8548f789a8fae/ch_PP-OCRv5_rec.onnx",
        "bf66820f48fa99f779974c4df78e5274a9d8e0458c4137e8c5357e40e2c3faf2",
    ),
    "dict_zhja.txt": (
        f"{HF}/webnn/PP-OCRv5-ONNX/resolve/006b4419a415072fbb16a1d7d0d8548f789a8fae/ch_PP-OCRv5_dict.txt",
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
    path = CACHE / name
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
            lines = src.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n").split("\n")
            (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"{name}: {len(lines)} characters")
            continue
        stem = name.removesuffix(".onnx")
        if args.fp32:
            shutil.copyfile(src, OUT / f"{stem}.fp32.onnx")
        print(f"{name}:")
        q = quantize_weights_int8(onnx.load(src))
        dst = OUT / f"{stem}.int8.onnx"
        onnx.save(q, dst)
        print(f"  {src.stat().st_size / 1e6:.1f} MB -> {dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
