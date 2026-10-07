"""WO02: user-diacritics preservation in the plane runtime.

Builds a REAL deterministic ONNX graph (no doubles): class_logits =
one_hot[(input_id + 7*plane_id) % TABLE], so K-pass cycling is fully
predictable and every assertion is on actual model output.
"""

from __future__ import annotations

import io

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TABLE = 512
N_CLASSES = 3
CLASSES = ["́", "̀", "́̀"]  # synthetic combining-mark classes


def _graph_bytes() -> bytes:
    table = np.zeros((TABLE, N_CLASSES), dtype=np.float32)
    table[np.arange(TABLE), np.arange(TABLE) % N_CLASSES] = 1.0
    init = numpy_helper.from_array(table, "table")
    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["B", "S"])
    plane = helper.make_tensor_value_info("plane_ids", TensorProto.INT64, ["B", "S"])
    out = helper.make_tensor_value_info("class_logits", TensorProto.FLOAT, ["B", "S", str(N_CLASSES)])
    idx = helper.make_node("Add", ["input_ids", "plane7"], ["sum"])
    mul = helper.make_node("Mul", ["sum", "plane7"], ["scaled"])
    mod = helper.make_node("Mod", ["scaled", "tablen"], ["idx"], fmod=0)
    gather = helper.make_node("Gather", ["table", "idx"], ["class_logits"], axis=0)
    graph = helper.make_graph(
        [idx, mul, mod, gather], "tiny_plane",
        [ids, plane], [out],
        initializer=[init,
                     numpy_helper.from_array(np.array(7, dtype=np.int64), "plane7"),
                     numpy_helper.from_array(np.array(TABLE, dtype=np.int64), "tablen")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    buf = io.BytesIO()
    onnx.save(model, buf)
    return buf.getvalue()


def make_model():
    from interscript.ml.plane import PlaneModel

    return PlaneModel(graph=_graph_bytes(), classes=CLASSES, k_passes=2)


def test_default_path_unchanged_by_flag():
    m = make_model()
    assert m.translate("abc") == m.translate("abc", preserve_diacritics=False)


def test_preserve_identity_fully_labeled():
    m = make_model()
    labeled = "áb̀"
    assert m.translate(labeled, preserve_diacritics=True) == labeled


def test_preserve_partial_keeps_user_classes_and_fills_rest():
    m = make_model()
    out = m.translate("a\u0301bc", preserve_diacritics=True)
    assert out.startswith("a\u0301")  # user class kept verbatim regardless of model cycle
    marks = "".join(CLASSES)
    assert "".join(ch for ch in out if ch not in marks) == "abc"  # bases in order


def test_preserve_leading_marks_roundtrip():
    m = make_model()
    labeled = "\u0301ab"
    out = m.translate(labeled, preserve_diacritics=True)
    assert out.startswith(labeled[0])  # leading mark kept verbatim
    assert "\x00" not in out  # anchor never leaks into output
    marks = "".join(CLASSES)
    assert "".join(ch for ch in out if ch not in marks) == "ab"


def test_preserve_multibyte_base_chars():
    m = make_model()
    labeled = "\u00e9\u0301"  # precomposed two-byte base + mark
    assert m.translate(labeled, preserve_diacritics=True) == labeled


def test_plain_text_still_translates():
    m = make_model()
    out = m.translate("abc")
    marks = "".join(CLASSES)
    assert set(out) <= set("abc") | set(marks)
    assert "".join(ch for ch in out if ch not in marks) == "abc"
    assert len(m.predict_token_classes("abc")) == 3
