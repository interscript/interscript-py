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


def test_unmapped_uppercase_passes_through_like_ruby():
    """Measured against the Ruby interpreter: there is NO implicit
    casing convention. With rules for Б Г А б г а but none for Я,
    БЯГА -> BЯGA (Я passes through), Бяга -> Byaga."""
    tree2 = parse_imp(
        'stage {\n  parallel {\n    sub "Б", "B"\n    sub "я", "ya"\n'
        '    sub "Г", "G"\n    sub "А", "A"\n    sub "г", "g"\n    sub "а", "a"\n  }\n}\n'
    )
    assert Engine(tree2).transliterate("БЯГА") == "BЯGA"
    assert Engine(tree2).transliterate("Бяга") == "Byaga"


@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_explicit_uppercase_rules_not_shadowed_by_implicit_casing():
    """bgnpcgn-ukr carries explicit sub "А" "A" rules; the engine's
    implicit casing anchors for lowercase rules shadowed them, so
    Авдіївська came out lowercase."""
    e = _load("bgnpcgn-ukr-Cyrl-Latn-1965")
    assert e.transliterate("Авдіївська") == "Avdiyivs’ka"
    assert e.transliterate("Міськрада") == "Mis’krada"


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


def test_boundary_uses_the_unicode_word_property_like_ruby():
    """Ruby's \\b is Unicode-aware over the Word property, which counts
    combining marks as word characters (its \\w is ASCII-only, but \\b
    does not follow \\w there). A word-final rule must not fire when a
    kasra follows the hamza carrier (dā'im, not dā'aim), and must fire
    at a true end of word."""
    tree = parse_imp(
        'stage {\n  parallel {\n'
        '    sub "ئ" + boundary, "\'a"\n'
        '    sub "ئ", "\'"\n'
        '    sub "ِ", "i"\n'
        '    sub "d", "d"\n  }\n}\n'
    )
    # ئ + kasra: the kasra is a Mark, hence a word char — no boundary,
    # and the bare-ئ rule fires, not the word-final one.
    assert Engine(tree).transliterate("dئِ") == "d'i"
    # ئ at a true end of word: the boundary rule fires.
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


def test_before_guard_over_alternation_of_anchor_and_literal():
    """odni-kor: before any([line_start, " "]) — a positive lookbehind
    over alternatives of DIFFERENT widths (0 and 1). Python re requires
    fixed-width lookbehinds; Ruby's Onigmo does not, so the branches are
    distributed: (?<=A|B) == (?:(?<=A)|(?<=B))."""
    import tempfile, os
    from interscript import add_load_path, transliterate

    d = tempfile.mkdtemp()
    open(os.path.join(d, "lb.isc"), "w").write(
        "system \"lb\" {\n"
        "  stage main {\n"
        "    parallel {\n"
        "      sub {\n"
        "        from \"x\"\n"
        "        to \"X\"\n"
        "        before any([line_start, \" \"])\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    add_load_path(d)
    assert transliterate("lb", "x") == "X"
    assert transliterate("lb", "a x") == "a X"
    assert transliterate("lb", "ax") == "ax"


def test_maybe_accepts_full_expressions():
    """alalc-aze / odni-pus: from maybe(any("ab")) + "c" — maybe over a
    character class, which the renderer rejected as non-string."""
    import tempfile, os
    from interscript import add_load_path, transliterate

    d = tempfile.mkdtemp()
    open(os.path.join(d, "mb.isc"), "w").write(
        "system \"mb\" {\n"
        "  stage main {\n"
        "    parallel {\n"
        "      sub {\n"
        "        from maybe(any(\"ab\")) + \"c\"\n"
        "        to \"Q\"\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    add_load_path(d)
    assert transliterate("mb", "c") == "Q"
    assert transliterate("mb", "ac") == "Q"
    assert transliterate("mb", "bc") == "Q"
    # empty maybe + "c" still matches the bare c (Ruby parallel
    # semantics: the rule consumes just "c" here).
    assert transliterate("mb", "dc") == "dQ"


@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_primitive_space_result_pads_the_string():
    """moct-kor pads with `sub line_start space` / `sub line_end space`;
    the subst renderer rejected primitive results and silently dropped
    the rules, so the before-space guards on initial consonants never
    fired: 불국사 -> ᄇulguksa instead of Bulguksa."""
    e = _load("moct-kor-Hang-Latn-2000")
    assert e.transliterate("불국사") == "Bulguksa"


@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_subst_boundary_treats_combining_marks_as_word_chars():
    """un-mar कंगना: the subst-family path used raw \\b, so the क|ं
    junction (anusvara is a combining mark) counted as a word boundary
    and the schwa-killing rule fired — kaṁganā came out kṁganā."""
    e = _load("un-mar-Deva-Latn-2016")
    assert e.transliterate("कंगना") == "kaṁganā"


def test_plain_parallel_duplicate_last_wins():
    """Measured against Ruby: an all-plain parallel block compiles to a
    replace tree where a later duplicate from overwrites an earlier one
    (masm-mon lists sub "i" "й" ... sub "i" "и" to let the last win).
    A guarded block takes the megaregexp path instead, where the
    earliest equal-key rule that matches wins."""
    tree = parse_imp(
        'stage {\n  parallel {\n'
        '    sub "i", "й"\n'
        '    sub "ab", "X"\n'
        '    sub "i", "и"\n'
        '    sub "ab", "Y"\n'
        '  }\n}\n'
    )
    e = Engine(tree)
    assert e.transliterate("i ab") == "и Y"

    tree2 = parse_imp(
        'stage {\n  parallel {\n'
        '    sub "i", "X", before: "a"\n'
        '    sub "i", "Y"\n'
        '  }\n}\n'
    )
    e2 = Engine(tree2)
    assert e2.transliterate("ai") == "aX"
    assert e2.transliterate("bi") == "bY"


def test_par_unsafe_rule_forces_megaregexp_first_wins():
    """Measured via Ruby on bgnpcgn-bal: one rule with a boundary in a
    par-unsafe position makes the whole block fall back from the
    replace tree (last duplicate wins) to the megaregexp (earliest
    equal-key rule wins) — و maps to o there, not the later w."""
    tree = parse_imp(
        'stage {\n  parallel {\n'
        '    sub "i", "X"\n'
        '    sub "i", "Y"\n'
        '    sub "z" + boundary, "B"\n'
        '  }\n}\n'
    )
    assert Engine(tree).transliterate("i") == "X"


@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_any_character_renders():
    """kp-kor hyphenates generics with before any_character +
    any_character guards; the item renderer had no branch for the
    any_character function item and silently dropped every such rule
    (고비리 -> Kobiri instead of Kobi-ri)."""
    e = _load("kp-kor-Hang-Latn-2002")
    assert e.transliterate("고비리") == "Kobi-ri"
    assert e.transliterate("교구동") == "Kyogu-dong"


@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_quote_escape_survives_the_round_trip():
    """gost-rus maps the hard sign to a double quote (to "\\""); the
    value round-trips through expression source, where the unescaper
    only decoded \\uXXXX — \\" stayed a backslash-quote and съезд came
    out s\\"ezd."""
    e = _load("gost-rus-Cyrl-Latn-7.79-2000-2002")
    assert e.transliterate("съезд") == 's"ezd'


@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_run_executes_the_named_dependency_stage():
    """mvd-rus-2010 runs the 2008 dependency's translit stage — not its
    main, which ends in compose. Running main precomposed l+U+0301 into
    ĺ before the postrule could strip the acute (Vasiĺeva)."""
    e = _load("mvd-rus-Cyrl-Latn-2010")
    assert e.transliterate("Васiльева") == "Vasileva"


@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_not_word_stdlib_alias_renders():
    """odni-che maps the digit 1 (a palochka stand-in) guarded by
    not_word; the alias had no stdlib entry and the rules were
    dropped — Ахмадк1ант kept its 1."""
    e = _load("odni-che-Cyrl-Latn-2015")
    assert e.transliterate("Ахмадк1ант") == "Akhmadkant"


@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_library_string_alias_is_a_class_inside_any():
    """Measured in Ruby: an alias imported from a library whose value
    is a plain string acts as a CHARACTER CLASS inside any() (the
    unicode library's greek), while an in-map string alias is a literal
    sequence. alalc-ell's γ-nasal rules guard on any(greek)."""
    e = _load("alalc-ell-Grek-Latn-2010")
    assert e.transliterate("γκέγκε") == "gkenke"
    assert e.transliterate("Λαγκαδάς") == "Lankadas"
