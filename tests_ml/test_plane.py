"""Tests for the plane model kind (run-018 on-device artifact).

The plane artifact ships one ONNX graph taking (input_ids, plane_ids)
to per-position class logits; decode is a host-side K-pass loop; the
haraqat plane renders onto the skeleton. Fixture graph: logits[i]
depends only on plane_ids[i] via class = (plane + 1) % n_classes, so
predictions cycle deterministically - enough to pin the loop, the
rendering, and the byte table.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
import numpy as np
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from interscript.ml.plane import PlaneModel, render_plane

N_CLASSES = 4
CLASSES = ["", "َ", "ُ", "ْ"]


def tiny_plane_graph() -> bytes:
    """class_logits[i, c] = 1 iff c == (plane_ids[i] + 1) % N_CLASSES
    (bias on the desired class so argmax is exact)."""
    init = numpy_helper.from_array(
        np.array(
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        ),
        "map",
    )
    node = helper.make_node("Gather", ["map", "plane_ids"], ["class_logits"], axis=0)
    graph = helper.make_graph(
        [node],
        "tiny-plane",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "t"]),
            helper.make_tensor_value_info("plane_ids", TensorProto.INT64, ["b", "t"]),
        ],
        outputs=[
            helper.make_tensor_value_info("class_logits", TensorProto.FLOAT, ["b", "t", "c"])
        ],
        initializer=[init],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    m.ir_version = 9
    return m.SerializeToString()


class TestRenderPlane(unittest.TestCase):
    def test_renders_classes_onto_skeleton(self):
        skel = "كتب"
        classes = ["", "َ", "ُ"]
        self.assertEqual(render_plane(skel, classes), "كتَبُ")

    def test_multibyte_chars_consume_one_class(self):
        self.assertEqual(render_plane("كتب", ["َ", "", ""]), "كَتب")


class TestPlaneModel(unittest.TestCase):
    def _model(self, k=2):
        return PlaneModel(graph=tiny_plane_graph(), classes=list(CLASSES), k_passes=k)

    def test_k_pass_loop_cycles_predictions(self):
        m = self._model(k=1)
        # one pass from MASK(=4): class = (4+1) % 4 = 1 -> fatha everywhere
        out = m.translate("كتب")
        self.assertEqual(out, "كَتَبَ")
        m2 = self._model(k=2)
        # second pass: 1 -> 2 -> damma
        self.assertEqual(m2.translate("كتب"), "كُتُبُ")

    def test_input_ids_follow_byte_table(self):
        m = self._model(k=1)
        ids = m.encode("ab")
        self.assertEqual(ids, [ord("a") + 3, ord("b") + 3])


if __name__ == "__main__":
    unittest.main()


class TestPlaneZipRoundTrip(unittest.TestCase):
    def _zip_bytes(self, tamper: bool = False) -> bytes:
        import hashlib
        import io
        import json
        import zipfile

        graph = tiny_plane_graph()
        classes_json = json.dumps(CLASSES, ensure_ascii=False).encode("utf-8")
        # metadata shas describe the AUTHENTIC members
        sha_g = hashlib.sha256(graph).hexdigest()
        sha_c = hashlib.sha256(classes_json).hexdigest()
        meta = (
            f"id: plane-fixture-1.0\nprecision: fp32\nkind: plane\nk_passes: 2\n"
            f"members:\n  plane.onnx: {sha_g}\n  classes.json: {sha_c}\n"
        )
        stored = bytearray(graph)
        if tamper:
            # valid-CRC corruption: stored content differs, metadata sha stale
            idx = stored.find(b"tiny-plane")
            stored[idx] ^= 0xFF
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("metadata.yaml", meta)
            z.writestr("plane.onnx", bytes(stored))
            z.writestr("classes.json", classes_json)
        return buf.getvalue()

    def test_round_trip_load(self):
        from interscript.ml.plane import PlaneModel

        m = PlaneModel.from_zip(self._zip_bytes())
        self.assertEqual(m.k, 2)
        self.assertEqual(m.translate("كتب"), "كُتُبُ")

    def test_tampered_graph_rejected(self):
        from interscript.ml.plane import PlaneModel

        with self.assertRaisesRegex(ValueError, "sha256"):
            PlaneModel.from_zip(self._zip_bytes(tamper=True))

    def test_missing_member_rejected(self):
        import io
        import zipfile

        from interscript.ml.plane import PlaneModel

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("metadata.yaml", "kind: plane\nmembers:\n  plane.onnx: " + "0" * 64 + "\n")
        with self.assertRaisesRegex(ValueError, "plane.onnx"):
            PlaneModel.from_zip(buf.getvalue())
