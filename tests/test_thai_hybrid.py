"""WO03: Thai lexicon fast-path + hybrid (maximal-munch, model fallback)."""

from __future__ import annotations

from interscript.ml.thai_hybrid import Lexicon, segment, hybrid_translate


def make_lexicon() -> Lexicon:
    return Lexicon({"กิน": "kin", "ข้าว": "kʰâːw", "กินข้าว": "kinkʰâːw"})


class FakeModel:
    def __init__(self, mark="|"):
        self.mark = mark
        self.calls: list[str] = []

    def translate(self, text: str, max_len: int = 256) -> str:
        self.calls.append(text)
        return self.mark + text + self.mark


def test_maximal_munch_prefers_longest_match():
    assert segment("กินข้าว", make_lexicon()) == [("กินข้าว", "kinkʰâːw")]


def test_segment_splits_dict_and_oov_runs():
    # "นอน" is OOV: the run extends until the next dictionary hit
    segs = segment("นอนกิน", make_lexicon())
    assert segs == [("นอน", None), ("กิน", "kin")]


def test_segment_preserves_spaces_and_punct():
    assert segment("กิน ข้าว!", make_lexicon()) == [
        ("กิน", "kin"), (" ", " "), ("ข้าว", "kʰâːw"), ("!", "!"),
    ]


def test_hybrid_uses_lexicon_then_model_for_oov():
    model = FakeModel()
    out = hybrid_translate("กินนอน", make_lexicon(), model)
    assert out == "kin|นอน|"
    assert model.calls == ["นอน"]  # one batched fallback span, not per char


def test_hybrid_all_dict_never_touches_model():
    model = FakeModel()
    out = hybrid_translate("กินข้าว", make_lexicon(), model)
    assert out == "kinkʰâːw"
    assert model.calls == []


def test_hybrid_all_oov_single_span():
    model = FakeModel()
    out = hybrid_translate("นอนๆ", make_lexicon(), model)
    assert out == "|นอนๆ|"
    assert model.calls == ["นอนๆ"]


class TestFetchLexicon:
    def test_fetches_sha_verified_and_builds_a_lexicon(self, tmp_path, monkeypatch):
        import json as _json

        lexicon = {"กิน": "kin", "ข้าว": "kʰâːw"}
        blob = _json.dumps(lexicon, ensure_ascii=False).encode("utf-8")
        import hashlib

        sha = hashlib.sha256(blob).hexdigest()
        (tmp_path / "lex.json").write_bytes(blob)
        base = f"file://{tmp_path}/lex.json"

        from interscript.ml import thai_hybrid as TH

        got = TH.fetch_lexicon(index_url=None, _base_url=base, _sha256=sha)
        assert got["กิน"] == "kin"
        assert isinstance(got, TH.Lexicon)

    def test_rejects_sha_mismatch(self, tmp_path):
        (tmp_path / "lex.json").write_bytes(b'{"a":"b"}')
        from interscript.ml import thai_hybrid as TH

        try:
            TH.fetch_lexicon(
                index_url=None,
                _base_url=f"file://{tmp_path}/lex.json",
                _sha256="0" * 64,
            )
        except ValueError as e:
            assert "sha256" in str(e)
        else:
            raise AssertionError("expected sha mismatch")
