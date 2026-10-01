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


def _load(name: str, on_unsupported: str = "raise") -> Engine:
    interscript.interscript._load_paths.clear()
    interscript.add_load_path(MAPS)
    return interscript.load_map(name, on_unsupported=on_unsupported)


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
    engine = _load("un-ell-Grek-Latn-1987-ts", on_unsupported="skip")
    cases = engine.tree["tests"]
    passed = sum(1 for src, want in cases if engine.transliterate(src) == want)
    assert passed >= 200, f"regression: {passed}/{len(cases)}"


def test_engine_separate_and_title_case():
    import tempfile, os, sys
    from interscript import add_load_path, transliterate

    d = tempfile.mkdtemp()
    open(os.path.join(d, "ops.isc"), "w").write(
        'system "ops" {\n'
        '  stage main {\n'
        "    separate separator \"|\"\n"
        "  }\n"
        "}\n"
    )
    add_load_path(d)
    assert transliterate("ops", "こんいちは") == "こ|ん|い|ち|は"

    open(os.path.join(d, "tc.isc"), "w").write(
        'system "tc" {\n'
        '  stage main {\n'
        "    title_case word_separator: \"\"\n"
        "  }\n"
        "}\n"
    )
    assert transliterate("tc", "hello world") == "Hello world"
    assert transliterate("tc", "hello world\nhello hello") == "Hello world\nHello hello"


def test_parallel_selection_matches_ruby_max_length():
    """Selection mirrors Ruby Rule::Sub#max_length: from + all guard
    lengths (+priority), zero-width aliases counting 1. A longer later
    rule (4) must beat an earlier boundary-guarded one (3) — the
    alalc-ara hamza shape ("\u0623\u064e" vs boundary+"\u0623")."""
    tree = parse_imp(
        'stage {\n  parallel {\n'
        '    sub boundary + "ab", "X"\n'
        '    sub "abcd", "Y"\n  }\n}\n'
    )
    assert Engine(tree).transliterate("abcd") == "Y"
    # Equal keys keep source order, like Ruby's index tiebreak.
    tree2 = parse_imp(
        'stage {\n  parallel {\n'
        '    sub "ab", "X"\n'
        '    sub "abc", "Y"\n  }\n}\n'
    )
    assert Engine(tree2).transliterate("abc") == "Y"


def test_boundary_treats_combining_marks_as_word_chars():
    """Ruby's \\b counts combining marks (Arabic diacritics) as word
    characters — a word-final rule must not fire when a kasra follows
    the hamza carrier (dā'im, not dā'aim)."""
    tree = parse_imp(
        'stage {\n  parallel {\n'
        '    sub "ئ" + boundary, "\'a"\n'
        '    sub "ئ", "\'"\n'
        '    sub "ِ", "i"\n'
        '    sub "d", "d"\n  }\n}\n'
    )
    # ئ + kasra: no boundary — the bare-ئ rule fires, not the final one.
    assert Engine(tree).transliterate("dئِ") == "d'i"
    # ئ at a true word end: the boundary rule fires.
    assert Engine(tree).transliterate("dئ") == "d'a"


@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_not_guards_are_match_constraints_not_sort_keys_only():
    """Ruby compiles not_before/not_after as real guards (negative
    lookbehind/lookahead — interpreter#build_regexp). The alalc-ara
    shape: mas'alah keeps the hamza mark only because أ+fatḥa -> 'a'
    declines when ة/ل FOLLOWS (not_after is following context)."""
    import tempfile, os
    from interscript import add_load_path, transliterate

    d = tempfile.mkdtemp()
    open(os.path.join(d, "ng.isc"), "w").write(
        "system \"ng\" {\n"
        "  stage main {\n"
        "    parallel {\n"
        "      sub {\n"
        "        from \"xy\"\n"
        "        to \"A\"\n"
        "        not_after any(\"ab\")\n"
        "      }\n"
        "      sub {\n"
        "        from \"zw\"\n"
        "        to \"B\"\n"
        "        not_before any(\"pq\")\n"
        "      }\n"
        "      sub \"x\" \"Q\"\n"
        "      sub \"z\" \"Z\"\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    add_load_path(d)
    # not_after: "xy" followed by a/b declines -> bare x rule fires.
    assert transliterate("ng", "xya") == "Qya"
    assert transliterate("ng", "xyc") == "Ac"
    # not_before: "zw" preceded by p/q declines.
    assert transliterate("ng", "pzw") == "pZw"
    assert transliterate("ng", "czw") == "cB"


def test_any_list_alternatives_are_full_expressions():
    """any([...]) alternatives are full items (Ruby: Any of Items, each
    may be a Group). The list tokenizer kept only quoted strings, so
    bare atoms like boundary were silently dropped and not_before
    guards lost their boundary branch — alalc-ara word-initial آ then
    took the medial "’ā" rule instead of "ā"."""
    tree = parse_imp(
        'stage {\n  parallel {\n'
        '    sub any([boundary + "ab", "q"]), "X"\n'
        '    sub "b", "Y"\n'
        '  }\n}\n'
    )
    e = Engine(tree)
    # boundary-guarded alternative fires at word start.
    assert e.transliterate("ab") == "X"
    assert e.transliterate("q") == "X"
    # no boundary before "ab" -> the alternative declines, bare b fires.
    assert e.transliterate("xab") == "xaY"
    assert e.transliterate("cab") == "caY"
