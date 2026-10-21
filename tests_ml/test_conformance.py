"""The frozen-fixture contract (interscript-models/conformance/) — the
python runtime must reproduce a model's vectors byte-exactly under the
declared decode mode, or the change does not merge.

Gated on env: SECRYST_E2E_ZIP (the artifact) + CONFORMANCE_VECTORS
(the fixture file), so CI runs without downloads."""

from __future__ import annotations

import json
import os

import pytest

ort = pytest.importorskip("onnxruntime")

from interscript.ml import Model  # noqa: E402


def test_conformance_vectors_byte_exact() -> None:
    vectors_path = os.environ.get("CONFORMANCE_VECTORS")
    zip_path = os.environ.get("SECRYST_E2E_ZIP")
    if not vectors_path or not zip_path:
        pytest.skip("set CONFORMANCE_VECTORS and SECRYST_E2E_ZIP")
    vectors = json.loads(open(vectors_path, encoding="utf-8").read())
    assert vectors["decode"]["mode"] == "single-row-greedy"
    model = Model.load(zip_path)
    failures = []
    for v in vectors["vectors"]:
        got = model.translate(v["src"])
        if got != v["out"]:
            failures.append(f"{v['src']!r} -> {got!r} (expected {v['out']!r})")
    assert not failures, (
        f"{vectors['model']}: {len(failures)}/{len(vectors['vectors'])} "
        f"mismatched\n" + "\n".join(failures[:3])
    )
