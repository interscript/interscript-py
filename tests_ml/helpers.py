"""Shared tiny-graph zip builder for runtime tests."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import yaml
from onnx import TensorProto, helper, numpy_helper

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


def _add_graph(name: str, inputs: list[str], output: str) -> bytes:
    graph = helper.make_graph(
        nodes=[helper.make_node("Add", [inputs[0], "bias"], [output])],
        name=name,
        inputs=[
            helper.make_tensor_value_info(n, TensorProto.INT64, ["batch", "seq"])
            for n in inputs
        ],
        outputs=[
            helper.make_tensor_value_info(output, TensorProto.INT64, ["batch", "seq"])
        ],
        initializer=[numpy_helper.from_array(np.zeros(1, dtype=np.int64), "bias")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=7
    )
    return model.SerializeToString()


def build_tiny_zip(
    path: Path, tamper: bool = False, manifest: dict | None = None
) -> Path:
    encoder = _add_graph("tiny-enc", ["input_ids"], "last_hidden_state")
    decoder = _add_graph(
        "tiny-dec", ["input_ids", "encoder_hidden_states"], "logits"
    )
    sha = {
        "encoder.onnx": hashlib.sha256(encoder).hexdigest(),
        "decoder.onnx": hashlib.sha256(decoder).hexdigest(),
    }
    if tamper:
        sha["encoder.onnx"] = "0" * 64
    meta = dict(manifest if manifest is not None else MANIFEST)
    meta["sha256"] = sha
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.yaml", yaml.safe_dump(meta))
        zf.writestr("encoder.onnx", encoder)
        zf.writestr("decoder.onnx", decoder)
        zf.writestr("README.md", "# tiny\n")
    return path
