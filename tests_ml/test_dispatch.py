"""Unified dispatch: transliterate(id, text) resolves map ids via the
engine and model ids via the ML layer (lazy, optional extra)."""

from __future__ import annotations

from pathlib import Path

import yaml

from interscript import transliterate


def _index_file(tmp_path: Path, zip_path: Path) -> str:
    import hashlib

    index = {
        "version": 1,
        "models": {
            "tiny-1.0": {
                "task": "translit",
                "precision": "fp32",
                "filename": zip_path.name,
                "url": f"file://{zip_path}",
                "sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                "size": zip_path.stat().st_size,
            }
        },
    }
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(index), encoding="utf-8")
    return str(path)


def test_transliterate_routes_model_id_through_ml_layer(tmp_path: Path) -> None:
    from interscript.ml import Model
    from tests_ml.helpers import build_tiny_zip

    zip_path = build_tiny_zip(tmp_path / "channel" / "tiny.zip")
    index = _index_file(tmp_path, zip_path)

    expected = Model.load(zip_path).translate("he", max_len=4)
    got = transliterate("tiny-1.0", "he", index_url=index)
    assert isinstance(got, str)
    assert got == expected


def test_transliterate_unknown_id_still_raises_file_not_found(tmp_path: Path) -> None:
    from tests_ml.helpers import build_tiny_zip

    index = _index_file(tmp_path, build_tiny_zip(tmp_path / "channel" / "tiny.zip"))
    try:
        transliterate("no-such-map-9.9", "he", index_url=index)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as e:
        assert "no-such-map-9.9" in str(e)


def test_transliterate_map_id_still_uses_engine(tmp_path: Path) -> None:
    import pytest

    from interscript import add_load_path

    maps = Path(__file__).resolve().parent.parent / "maps"
    if not maps.exists():
        pytest.skip("maps corpus not checked out")
    add_load_path(str(maps))
    out = transliterate("bgnpcgn-rus-cyrl-latn-2016", "Москва")
    assert isinstance(out, str) and out
