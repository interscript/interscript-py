"""ISC-WF-1 construct coverage — interscript-py#47.

The 22-map parse-failure classes fixed against the trio's accept
behavior (rs/ruby/ts all parse these maps; the py parser lacked the
constructs, so 435/457 corpus maps loaded):

- ``languages`` declarations + ``when`` rule guards (the iso-*-15919-2026
  family; W3 selection semantics)
- ``ruleset``/``use``/``override`` expansion (the Thai maps splice a
  dependency ruleset; 8 further maps had their ``use`` lines silently
  skipped, diverging from the trio's spliced trees)
- free-text ``note`` fallback in metadata notes blocks (din-urd,
  komaba — the reference grammars' ordered choice: note-line list, then
  generic-field raw text)
- the ``rababa`` funcall loading (execution stays a named unsupported
  op, mirroring ruby's ExternalUtilError-without-the-gem contract)

Inline sources pin the grammar; corpus fixtures (the CI checkout
provides interscript/maps) pin the actual failing maps and replay the
golden corpus byte-for-byte.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from interscript.engine import Engine, ExecutionError
from interscript.isc import IscParseError, UnsupportedConstruct, isc_to_tree

MAPS = Path(
    os.environ.get("INTERSCRIPT_MAPS_PATH", Path(__file__).parent.parent.parent / "maps" / "maps")
)
LIBS = MAPS.parent / "libs"
GOLDEN = Path(__file__).parent / "golden" / "isc-wf0.jsonl"

corpus = pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")


def _parse(source: str) -> dict:
    return isc_to_tree(source, "test.isc")


# ---------------------------------------------------------------------------
# Class A — `languages` declarations + `when` guards (W3)
# ---------------------------------------------------------------------------


def test_languages_decl_keyword_guarded_list():
    """`languages a b, c` — comma or whitespace separated, the list stops
    at the next block keyword instead of swallowing it."""
    doc = _parse(
        'system "x" {\n'
        "languages opt98 opt99, opt10\n"
        "tests { }\n"
        "stage main { parallel { sub \"a\" \"b\" } }\n"
        "}\n"
    )
    assert doc["languages"] == ["opt98", "opt99", "opt10"]


def test_languages_decl_requires_identifier():
    with pytest.raises(IscParseError):
        _parse('system "x" {\nlanguages\nstage main { parallel { sub "a" "b" } } }\n')


@corpus
def test_iso_15919_maps_declare_languages():
    """The 17 iso-*-15919-2026 maps failed at `languages` before; every
    one of them parses and carries its declaration."""
    expected = {
        "iso-ben-Beng-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-san-Deva-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-tam-Taml-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-doi-Takr-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-guj-Gujr-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-kan-Kana-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-kas-Shrd-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-mal-Mlym-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-mar-Modi-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-new-Newa-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-ori-Orya-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-pan-Guru-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-pra-Brah-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-sin-Sinh-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-tel-Telu-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-inc-Kthi-Latn-15919-2026": ["opt98", "opt99", "opt10"],
        "iso-san-Gran-Latn-15919-2026": ["opt98", "opt99", "opt10"],
    }
    for name, languages in expected.items():
        from interscript.isc import parse_isc_file

        tree = parse_isc_file(MAPS / f"{name}.isc")
        assert tree.get("languages") == languages, name


def test_when_guard_selection_semantics():
    """§5.1/§5.2: guards are inert without a selection; a declared
    selection activates the guarded rule and it REPLACES the unguarded
    rule sharing its from; an undeclared selection is inert."""
    src = (
        'system "x" {\n'
        "languages kur\n"
        'stage main { parallel { sub "a" "1" sub "a" "2" when kur } }\n'
        "}\n"
    )
    engine = Engine(_parse(src))
    assert engine.transliterate("aaa") == "111"
    assert engine.transliterate("aaa", language="kur") == "222"
    # undeclared selection: inert, like none
    assert engine.transliterate("aaa", language="zzz") == "111"


@corpus
def test_iso_15919_default_and_selected_transliteration():
    """iso-ben: the unguarded row applies by default, the `when opt98`
    row only under the declared selection (the map's own tests pin the
    default; the opt98 reading comes from clause 9.8 of the spec)."""
    import interscript

    interscript.add_load_path(str(LIBS))
    assert interscript.transliterate("iso-ben-Beng-Latn-15919-2026", "অ") == "a"
    assert (
        interscript.transliterate("iso-ben-Beng-Latn-15919-2026", "অ", language="opt98")
        == "A"
    )


# ---------------------------------------------------------------------------
# Class B — `ruleset`/`use`/`override` expansion (W1/W2)
# ---------------------------------------------------------------------------


def test_ruleset_use_override_same_map():
    """W2 child-wins on exact from-equality; unmatched overrides are
    added; the override rows join a trailing parallel block (§4.2.5) —
    one longest-match pass over base + overrides."""
    src = (
        'system "x" {\n'
        "ruleset digraphs {\n"
        '  parallel { sub "ab" "1" sub "cd" "2" }\n'
        "}\n"
        'stage main { use digraphs override { sub "ab" "X" sub "ef" "Y" } }\n'
        "}\n"
    )
    engine = Engine(_parse(src))
    # "ab" -> "X" (override wins), "cd" -> "2" (kept), "ef" -> "Y" (added)
    assert engine.transliterate("abcd") == "X2"
    assert engine.transliterate("efab") == "YX"


def test_use_without_override_splices_ruleset():
    src = (
        'system "x" {\n'
        "ruleset rows {\n"
        '  sub "a" "1"\n'
        '  parallel { sub "bb" "2" }\n'
        "}\n"
        "stage main { use rows }\n"
        "}\n"
    )
    engine = Engine(_parse(src))
    assert engine.transliterate("abb") == "12"


def test_use_in_parallel_row_flattens_into_the_pass():
    """§3.2: a `use` row inside a parallel block flattens the referenced
    content into that single pass."""
    src = (
        'system "x" {\n'
        "ruleset extra {\n"
        '  sub "c" "C"\n'
        "}\n"
        'stage main { parallel { sub "ab" "1" use extra } }\n'
        "}\n"
    )
    engine = Engine(_parse(src))
    assert engine.transliterate("abc") == "1C"


def test_use_cycle_and_unknown_ruleset_are_named_load_errors():
    cycle = (
        'system "x" {\n'
        "ruleset a { use b }\n"
        "ruleset b { use a }\n"
        "stage main { use a }\n"
        "}\n"
    )
    with pytest.raises(UnsupportedConstruct, match="cycle"):
        _parse(cycle)
    unknown = 'system "x" { stage main { use nosuch } }\n'
    with pytest.raises(UnsupportedConstruct, match="ruleset not found"):
        _parse(unknown)


def test_duplicate_ruleset_declaration_is_a_named_load_error():
    """WO18 G7 (py canon): `ruleset a` declared twice is a named load
    error — keep-first (ruby/ts) and last-wins/merge (rs pre-fix, py
    0.2.x) silently hide map bugs and make the executed body
    declaration-order dependent. ruby (MapLogicError), ts (IscLoadError)
    and rs (MapLogicError) reject; the 0.2.x parser rewrite lost the
    0.1.0 check. Spec §6 item 4's load-error surface for cycles,
    extended to duplicate definition (maps docs/parity G7)."""
    block_use = (
        'system "x" {\n'
        "ruleset a {\n"
        "  sequence { sub \"x\" \"1\" }\n"
        "}\n"
        "ruleset a {\n"
        "  sequence { sub \"z\" \"3\" }\n"
        "}\n"
        "stage main {\n"
        "  sequence { use a }\n"
        "}\n"
        "}\n"
    )
    stage_use = (
        'system "x" {\n'
        "ruleset a {\n"
        "  parallel { sub \"x\" \"1\" }\n"
        "}\n"
        "ruleset a {\n"
        "  parallel { sub \"z\" \"3\" }\n"
        "}\n"
        "stage main { use a }\n"
        "}\n"
    )
    for src in (block_use, stage_use):
        with pytest.raises(UnsupportedConstruct, match='ruleset "a" declared more than once'):
            _parse(src)


@corpus
def test_thai_maps_load_and_splice_the_dependency_ruleset():
    """alalc-tha-2011 and bgnpcgn-tha-2002 failed at `parallel` before:
    py skipped the `use ... override {}` line, the override block's rules
    leaked into the stage, and the override's closing brace closed the
    stage early. Both splice libs/thai-latn.isc's ruleset now; the
    golden rows below pin the trio's outputs."""
    import interscript

    interscript.add_load_path(str(LIBS))
    # alalc-tha golden rows (macron overrides REPLACE base rows)
    assert interscript.transliterate("alalc-tha-Thai-Latn-2011", "กำ") == "kam"
    assert interscript.transliterate("alalc-tha-Thai-Latn-2011", "ไทย") == "thai"
    assert interscript.transliterate("alalc-tha-Thai-Latn-2011", "นคร") == "nkhǭn"
    # bgnpcgn-tha golden rows (overrides ADD new rows)
    assert interscript.transliterate("bgnpcgn-tha-Thai-Latn-2002", "ภาษาไทย") == "phasathai"
    assert interscript.transliterate("bgnpcgn-tha-Thai-Latn-2002", "ใหม่") == "mai"
    assert interscript.transliterate("bgnpcgn-tha-Thai-Latn-2002", "เขา") == "khao"


@corpus
def test_previously_skipped_use_lines_splice_now():
    """8 maps loaded with their `use` lines silently DROPPED — trees that
    diverged from the trio. They load and transliterate their embedded
    tests now (the alalc-jba output requires the hebr-latn splices)."""
    import interscript

    interscript.add_load_path(str(LIBS))
    for name in (
        "alalc-jba-Hebr-Latn-2011",
        "alalc-jav-Arab-Latn-2012",
        "alalc-jav-Java-Latn-2012",
        "alalc-lad-Hebr-Latn-2025",
        "alalc-zsm-Arab-Latn-2011",
        "iso-ara-Arab-Latn-233-2026",
        "iso-ara-Arab-Latn-233-2026-modified",
        "odni-ara-Arab-Latn-2015",
    ):
        engine = interscript.load_map(name)
        assert engine.tree["stages"], name
        for input_text, expected in engine.tree["tests"][:3]:
            assert engine.transliterate(input_text) == expected, (name, input_text)


# ---------------------------------------------------------------------------
# Class C — free-text notes (ordered-choice fallback)
# ---------------------------------------------------------------------------


def test_notes_free_text_fallback_on_unescaped_quotes():
    """The note-line production fails on prose with unescaped inner
    quotes; the reference grammars fall back to one raw-text value for
    the whole block (ruby: notes = Slice 'note "...')."""
    doc = _parse(
        'system "x" {\n'
        "metadata {\n"
        '  notes {\n'
        '    note "heh "usually written" form"\n'
        "  }\n"
        "}\n"
        "}\n"
    )
    assert doc["metadata"]["notes"] == 'note "heh "usually written" form"'


def test_notes_line_list_still_yields_a_list():
    doc = _parse(
        'system "x" {\n'
        "metadata {\n"
        "  notes {\n"
        '    note "one"\n'
        '    note "two"\n'
        "  }\n"
        "}\n"
        "}\n"
    )
    assert doc["metadata"]["notes"] == ["one", "two"]


@corpus
def test_din_urd_and_komaba_notes_are_raw_text():
    """din-urd (unescaped quotes in the note prose) and komaba (a
    ruby-ish `% (APOS, ...)` suffix) failed at `note` before; the trio
    read both blocks as one raw text value."""
    from interscript.isc import parse_isc_file

    for name, head in (
        ("din-urd-Arab-Latn-31635-2011", 'note "Waw (row 34)'),
        ("var-jpn-Hrkt-Latn-komaba-2009", 'note "Syllabic n'),
    ):
        tree = parse_isc_file(MAPS / f"{name}.isc")
        notes = tree["metadata"]["notes"]
        assert isinstance(notes, str), name
        assert notes.startswith(head), name


# ---------------------------------------------------------------------------
# Class D — the rababa funcall loads; execution names the callee
# ---------------------------------------------------------------------------


@corpus
def test_rababa_funcall_loads_and_names_at_exec():
    import interscript

    engine = interscript.load_map("var-ara-Arab-Arab-rababa")
    with pytest.raises(ExecutionError, match="funcall rababa"):
        engine.transliterate("قطر")
    # skip mode: the engine records the named op at exec time and moves on
    tree = isc_to_tree(
        (MAPS / "var-ara-Arab-Arab-rababa.isc").read_text(encoding="utf-8"),
        "rababa.isc",
    )
    skipper = Engine(tree, on_unsupported="skip")
    assert skipper.transliterate("abc") == "abc"
    assert any("funcall rababa" in what for what in skipper.skipped_unsupported)


# ---------------------------------------------------------------------------
# Whole-corpus proof: 457/457 parse + the golden replay
# ---------------------------------------------------------------------------


@corpus
def test_full_corpus_parses():
    """Every corpus map parses (457/457 — the issue's headline number)."""
    from interscript.isc import parse_isc_file

    files = sorted(MAPS.glob("*.isc"))
    assert len(files) == 457
    for path in files:
        parse_isc_file(path)


corpus_and_golden = pytest.mark.skipif(
    not (MAPS.is_dir() and GOLDEN.is_file()),
    reason="interscript maps repo / golden corpus not present",
)


@corpus_and_golden
def test_golden_corpus_replay():
    """tests/golden/isc-wf0.jsonl (the rs golden corpus, 16,322 rows over
    453 maps) replays byte-identically, with no exclusions.

    The former exclusion (all rows of gki-bel-2000 / mvd-bel-2008 /
    odni-bel-2015) is gone: the capital-``\\U`` escape now decodes
    4..8 hex digits (esc_uni, the WO19 G6 ruling both spellings
    decode), so the three maps' uppercase-Ў rules fire and the corpus
    converges 16,322/16,322.
    """
    import interscript

    interscript.add_load_path(str(LIBS))
    rows = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    engines: dict[str, Engine] = {}
    mismatches = []
    for row in rows:
        engine = engines.get(row["map"])
        if engine is None:
            engines[row["map"]] = engine = interscript.load_map(row["map"])
        got = engine.transliterate(row["src"])
        if got != row["out"]:
            mismatches.append((row["map"], row["src"], row["out"], got))
    assert not mismatches, mismatches[:10]
