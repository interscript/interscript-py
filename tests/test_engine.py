from pathlib import Path

import pytest

import interscript
from interscript.engine import Engine, ExecutionError
from interscript.parser import parse_imp

SAMPLE = """
metadata {
  authority_id: test
  id: 1
  name: sample
}

tests {
  test "hello", "HELLO"
}

stage {
  parallel {
    sub "hello", "HELLO"
  }
}
"""

MAPS = Path(__file__).parent.parent.parent / "maps" / "maps"
import os
if os.environ.get("INTERSCRIPT_MAPS_PATH"):
    MAPS = Path(os.environ["INTERSCRIPT_MAPS_PATH"])


def _load(name: str) -> Engine:
    interscript._load_paths.clear()
    interscript.add_load_path(MAPS)
    return interscript.load_map(name)


def test_parse_metadata_and_tests():
    tree = parse_imp(SAMPLE)
    assert tree["metadata"]["authority_id"] == "test"
    assert ("hello", "HELLO") in tree["tests"]


def test_parallel_substitution():
    engine = Engine(parse_imp(SAMPLE))
    assert engine.transliterate("say hello") == "say HELLO"


def test_all_caps_word_uppercases_result():
    tree2 = parse_imp(
        'stage {\n  parallel {\n    sub "Б", "B"\n    sub "я", "ya"\n'
        '    sub "Г", "G"\n    sub "А", "A"\n    sub "г", "g"\n    sub "а", "a"\n  }\n}\n'
    )
    assert Engine(tree2).transliterate("БЯГА") == "BYAGA"
    assert Engine(tree2).transliterate("Бяга") == "Byaga"


def test_unsupported_construct_raises_by_default():
    tree = parse_imp('stage {\n  secryst "model"\n}\n')
    with pytest.raises(ExecutionError):
        Engine(tree).transliterate("x")


@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_real_bulgarian_map_full_parity():
    """98/98 of the map's own embedded tests — measured 2026-08-19."""
    engine = _load("un-bul-Cyrl-Latn-1977")
    engine.on_unsupported = "skip"  # map has 3 contextual subs, unused by its tests
    cases = engine.tree["tests"]
    fails = [(s, w, engine.transliterate(s)) for s, w in cases if engine.transliterate(s) != w]
    assert not fails, f"{len(fails)}/{len(cases)} failed; first: {fails[:2]}"


@pytest.mark.xfail(reason="loads and runs, but titlecase-of-proper-noun results + diphthong context rules pending", strict=False)
@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_real_persian_map_runs():
    engine = _load("bgnpcgn-prs-Arab-Latn-2007")
    engine.on_unsupported = "skip"
    engine.transliterate("بَغْلان")


@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_real_greek_map_dependency_runs():
    """Dependency-aliased dotted runs resolve; measured 206/242 embedded
    tests (2026-08-20). Remaining failures: letter-class context guards
    from the posix aliases."""
    engine = _load("un-ell-Grek-Latn-1987-ts")
    engine.on_unsupported = "skip"
    cases = engine.tree["tests"]
    passed = sum(1 for src, want in cases if engine.transliterate(src) == want)
    assert passed >= 200, f"regression: {passed}/{len(cases)}"
