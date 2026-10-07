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
