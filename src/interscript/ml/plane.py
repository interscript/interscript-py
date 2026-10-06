"""Plane model kind: the run-018 on-device artifact family.

One ONNX graph maps (input_ids, plane_ids) -> per-position class
logits; decode is a host-side K-pass loop (pass k>1 conditions on
pass k-1 argmax); the haraqat plane renders onto the skeleton.
Byte table: id = byte + 3 (the canonical ByT5 table); MASK id =
len(classes).
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

MASK_OFFSET = 0  # plane vocab = classes + MASK at index len(classes)


def render_plane(skeleton: str, char_classes: list[str]) -> str:
    """One class per CHARACTER (multi-byte chars consume one class)."""
    return "".join(ch + cls for ch, cls in zip(skeleton, char_classes))


class PlaneModel:
    def __init__(self, graph: bytes, classes: list[str], k_passes: int = 2):
        self.classes = classes
        self.n_classes = len(classes)
        self.mask_id = self.n_classes
        self.k = k_passes
        self.sess = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])

    def encode(self, text: str) -> list[int]:
        return [b + 3 for b in text.encode("utf-8")]

    def predict_token_classes(self, text: str) -> list[int]:
        ids = np.array([self.encode(text)], dtype=np.int64)
        plane = np.full_like(ids, self.mask_id)
        for _ in range(self.k):
            logits = self.sess.run(None, {"input_ids": ids, "plane_ids": plane})[0]
            plane = logits.argmax(-1)
        return plane[0].tolist()

    def translate(self, text: str) -> str:
        tok_preds = self.predict_token_classes(text)
        # majority vote of a char's byte-token predictions
        pos = 0
        char_classes: list[str] = []
        for ch in text:
            n = len(ch.encode("utf-8"))
            votes = tok_preds[pos : pos + n]
            cid = max(set(votes), key=votes.count)
            char_classes.append(self.classes[cid] if cid < self.n_classes else "")
            pos += n
        return render_plane(text, char_classes)


def from_zip(data: bytes) -> PlaneModel:
    """Load a plane artifact zip: metadata.yaml (id, precision, kind,
    k_passes, classes, member sha256s) + plane.onnx + classes.json.
    Every member is sha256-verified before the session is created."""
    import hashlib
    import io
    import json as _json
    import zipfile

    import yaml

    zf = zipfile.ZipFile(io.BytesIO(data))
    names = set(zf.namelist())
    if "metadata.yaml" not in names:
        raise ValueError("plane zip: metadata.yaml missing")
    meta = yaml.safe_load(zf.read("metadata.yaml"))
    if meta.get("kind") != "plane":
        raise ValueError(f"plane zip: kind={meta.get('kind')!r}, expected 'plane'")
    members = meta.get("members") or {}
    for member in ("plane.onnx", "classes.json"):
        if member not in names:
            raise ValueError(f"plane zip: {member} missing")
        want = members.get(member)
        got = hashlib.sha256(zf.read(member)).hexdigest()
        if want is not None and want != got:
            raise ValueError(f"plane zip: {member} sha256 mismatch ({got})")
        if want is None:
            raise ValueError(f"plane zip: {member} has no sha256 in metadata")
    classes = _json.loads(zf.read("classes.json"))
    return PlaneModel(
        graph=zf.read("plane.onnx"),
        classes=classes,
        k_passes=int(meta.get("k_passes", 2)),
    )


PlaneModel.from_zip = staticmethod(from_zip)
