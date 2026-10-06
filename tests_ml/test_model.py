"""Tests for the interscript-ml runtime.

Tiny-graph zips built with the onnx package (no torch, no training
repo). The end-to-end golden test runs only when a real zip is provided
via SECRYST_E2E_ZIP.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest
import yaml

ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from interscript.ml import Model, ModelFormatError, decode, encode  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

import numpy as np  # noqa: E402


def _graph(opset: int = 14) -> bytes:
    graph = helper.make_graph(
        nodes=[helper.make_node("Add", ["input_ids", "bias"], ["last_hidden_state"])],
        name="tiny-enc",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])
        ],
        outputs=[
            helper.make_tensor_value_info(
                "last_hidden_state", TensorProto.INT64, ["batch", "seq"]
            )
        ],
        initializer=[numpy_helper.from_array(np.zeros(1, dtype=np.int64), "bias")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=7
    )
    return model.SerializeToString()


def _decoder_graph() -> bytes:
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Add", ["input_ids", "bias"], ["logits"])
        ],
        name="tiny-dec",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
            helper.make_tensor_value_info(
                "encoder_hidden_states", TensorProto.INT64, ["batch", "seq"]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.INT64, ["batch", "seq"])
        ],
        initializer=[numpy_helper.from_array(np.zeros(1, dtype=np.int64), "bias")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=7
    )
    return model.SerializeToString()


MANIFEST = {
    "format": "imf-v1",
    "id": "tiny-1.0",
    "task": "translit",
    "source_script": "Latn",
    "target": "Latn",
    "tokenizer": "bytes",
    "opset": 14,
    "decoder": "plain",
    "precision": "fp32",
    "license": "BSD-3-Clause",
    "trained_from": "runtime test fixture",
}


def _tiny_zip(path: Path, tamper: bool = False, manifest: dict | None = None) -> Path:
    encoder, decoder = _graph(), _decoder_graph()
    sha = {
        "encoder.onnx": hashlib.sha256(encoder).hexdigest(),
        "decoder.onnx": hashlib.sha256(decoder).hexdigest(),
    }
    if tamper:
        sha["encoder.onnx"] = "0" * 64
    meta = dict(manifest if manifest is not None else MANIFEST)
    meta["sha256"] = sha
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.yaml", yaml.safe_dump(meta))
        zf.writestr("encoder.onnx", encoder)
        zf.writestr("decoder.onnx", decoder)
        zf.writestr("README.md", "# tiny\n")
    return path


def test_token_table() -> None:
    assert encode("rok") == [117, 114, 110, 1]
    assert decode([117, 114, 110]) == "rok"
    assert decode([117, 1, 114]) == "r"
    assert decode([]) == ""


def test_load_and_decode_tiny(tmp_path: Path) -> None:
    z = _tiny_zip(tmp_path / "tiny.zip")
    model = Model.load(z)
    assert model.id == "tiny-1.0"
    # tiny graphs are identity Adds: logits echo the decoder prefix, so
    # greedy emits encode(PAD-prefix input)+... — deterministic, not
    # meaningful; what matters is that the loop runs and decodes.
    tokens = model.generate("he", max_len=4)
    assert isinstance(tokens, list)
    text = model.translate("he", max_len=4)
    assert isinstance(text, str)


def test_sha256_mismatch_rejected(tmp_path: Path) -> None:
    z = _tiny_zip(tmp_path / "bad.zip", tamper=True)
    with pytest.raises(ModelFormatError, match="sha256 mismatch"):
        Model.load(z)


def test_non_bytes_tokenizer_rejected(tmp_path: Path) -> None:
    manifest = dict(MANIFEST, tokenizer="sentencepiece")
    z = _tiny_zip(tmp_path / "spm.zip", manifest=manifest)
    with pytest.raises(ModelFormatError, match="byte-level only"):
        Model.load(z)


def test_missing_graph_rejected(tmp_path: Path) -> None:
    z = _tiny_zip(tmp_path / "m.zip")
    truncated = tmp_path / "trunc.zip"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(truncated, "w") as dst:
        for name in src.namelist():
            if name != "decoder.onnx":
                dst.writestr(name, src.read(name))
    with pytest.raises(ModelFormatError, match="decoder.onnx"):
        Model.load(truncated)


def test_golden_set_e2e() -> None:
    """Run against a real zip: byte-identical outputs on the golden set."""
    zip_path = os.environ.get("SECRYST_E2E_ZIP")
    if not zip_path:
        pytest.skip("set SECRYST_E2E_ZIP to a real IMF zip")
    golden = Path(__file__).resolve().parent.parent.parent / "golden" / "khm-latn-100.jsonl"
    if "khm" not in Path(zip_path).name:
        pytest.skip("golden file is khm-latn specific")
    model = Model.load(zip_path)
    rows = [json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        assert model.translate(row["input"], max_len=128) == row["output"], row["input"]
