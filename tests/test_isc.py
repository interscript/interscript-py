"""Unit tests for the ISC parser and converter (interscript.isc)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from interscript.isc import IscParseError, UnsupportedConstruct, isc_to_tree


SAMPLE = '''
system "test-map-Cyrl-Latn-2000" {

metadata {
  name "Test Map"
  description {
    A description with { nested } braces.
  }
}

dependency "other-map" as dep

aliases {
  vowel = any("aeiou")
  upper_vowel = any("A".."Z")
}

stage main {
  run map.dep.stage.main

  parallel {
    sub "а" "a"
    sub {
      from "б" + "в"
      to "bv"
    }
    sub {
      from "г" + vowel
      to "h" + vowel
      before any("з")
    }
  }

  sequence {
    sub "дд" "d"
  }

  sub "еe" "ee"
}

tests {
  "абв" -> "abv"
  "гg" -> "hg" note "context note"
}
}
'''


def test_system_code_and_metadata():
    tree = isc_to_tree(SAMPLE)
    assert tree["metadata"]["name"] == "Test Map"
    assert "nested" in tree["metadata"]["description"]


def test_tests_render_as_pairs():
    tree = isc_to_tree(SAMPLE)
    assert tree["tests"][0] == ("абв", "abv")


def test_dependencies_resolve_alias_for_run():
    tree = isc_to_tree(SAMPLE)
    stage_children = tree["stages"][0]["children"]
    assert stage_children[0] == {"kind": "run", "map": "other-map", "stage": "main"}


def test_parallel_subs_render_as_expr_source():
    tree = isc_to_tree(SAMPLE)
    parallel = [c for c in tree["stages"][0]["children"] if c["kind"] == "parallel"][0]
    assert parallel["subs"][0] == {"pattern": '"а"', "result": '"a"'}
    assert parallel["subs"][1]["pattern"] == '"б" + "в"'


def test_sequence_and_bare_rules_render_as_regex():
    tree = isc_to_tree(SAMPLE)
    subs = [c for c in tree["stages"][0]["children"] if c["kind"] == "subst"]
    assert any(s["pattern"] == "дд" and s["result"] == "d" for s in subs)
    assert any(s["pattern"] == "еe" and s["result"] == "ee" for s in subs)


def test_aliases_inline_into_constraints():
    tree = isc_to_tree(SAMPLE)
    parallel = [c for c in tree["stages"][0]["children"] if c["kind"] == "parallel"][0]
    constrained = parallel["subs"][2]
    assert constrained["pattern"] == '"г" + any("aeiou")'
    assert constrained["before"] == "any(" + '"з"' + ")"


def test_range_alias_renders():
    tree = isc_to_tree(SAMPLE)
    # upper_vowel is referenced nowhere; aliases alone must not fail.
    assert tree["metadata"]["name"]


def test_parse_error_carries_position():
    with pytest.raises(IscParseError, match="line 1"):
        isc_to_tree('garbage {"')


def test_unsupported_construct_raises_by_default():
    bad = 'system "x" { stage main { parallel { sub { from "a" to "b" before unresolved_alias } } } }'
    with pytest.raises(UnsupportedConstruct, match="unresolved"):
        isc_to_tree(bad)


def test_capture_in_parallel_degrades_to_subst():
    bad = 'system "x" { stage main { parallel { sub { from capture("a") to "b" } } } }'
    tree = isc_to_tree(bad)
    children = tree["stages"][0]["children"]
    assert children[0]["kind"] == "subst"
    assert children[0]["pattern"] == "(a)"


def test_skip_mode_records_and_drops():
    bad = 'system "x" { stage main { parallel { sub "a" "b" } sequence { sub "c" "d" } } }'
    tree = isc_to_tree(bad, on_unsupported="skip")
    assert tree["skipped_unsupported"] == []
    tree2 = isc_to_tree(
        'system "x" { stage main { sequence { sub { from "c" to "d" before unresolved_alias } } } }',
        on_unsupported="skip",
    )
    assert tree2["skipped_unsupported"]
    assert tree2["stages"][0]["children"] == []


def test_to_position_case_function():
    src = 'system "x" { stage main { sub { from "b" to upcase } } }'
    tree = isc_to_tree(src)
    subst = tree["stages"][0]["children"][0]
    assert subst["kind"] == "subst"
    assert subst["case"] == "upcase"


def test_same_map_stage_runs_inline():
    src = """
    system "x" {
      stage one { sub "0" "1" }
      stage two { sub "1" "2" }
      stage main {
        run stage.one
        run stage.two
      }
    }
    """
    tree = isc_to_tree(src)
    main = next(s for s in tree["stages"] if s["name"] == "main")
    # Inlined in run order: both referenced stages' rules, self-contained.
    assert [c["result"] for c in main["children"]] == ["1", "2"]


def test_cyclic_stage_run_raises():
    src = """
    system "x" {
      stage a { run stage.b }
      stage b { run stage.a }
      stage main { run stage.a }
    }
    """
    with pytest.raises(UnsupportedConstruct, match="cyclic"):
        isc_to_tree(src)


def test_imported_stage_run_resolves_to_dependency():
    src = """
    system "x" {
      dependency "imp"
      stage main { run stage.hello }
    }
    """
    tree = isc_to_tree(src)
    assert tree["stages"][0]["children"][0] == {"kind": "run", "map": "imp"}


def test_separate_stage_op():
    src = 'system "x" { stage main { separate separator "|" } }'
    tree = isc_to_tree(src)
    assert tree["stages"][0]["children"][0] == {"kind": "separate", "separator": "|"}


def test_title_case_word_separator():
    src = 'system "x" { stage main { title_case word_separator: "" } }'
    tree = isc_to_tree(src)
    assert tree["stages"][0]["children"][0] == {
        "kind": "title_case",
        "word_separator": "",
    }


def test_any_in_result_picks_first_alternative():
    src = 'system "x" { stage main { sub any("ab") any(["XA", "YB"]) } }'
    tree = isc_to_tree(src)
    subst = tree["stages"][0]["children"][0]
    assert subst["result"] == "XA"


def test_any_in_result_nested_first():
    src = 'system "x" { stage main { sub "a" any([any("XY"), "Z"]) } }'
    tree = isc_to_tree(src)
    assert tree["stages"][0]["children"][0]["result"] == "X"


def test_any_in_result_none_first():
    src = 'system "x" { stage main { sub "a" any([none, " "]) } }'
    tree = isc_to_tree(src)
    assert tree["stages"][0]["children"][0]["result"] == ""








def _write_dep_fixture():
    import tempfile, os

    from interscript import add_load_path

    d = tempfile.mkdtemp()
    open(os.path.join(d, "dep-map.isc"), "w").write(
        'system "dep-map" {\n'
        "  aliases {\n"
        '    from_name = "404"\n'
        '    to_name = "500"\n'
        "  }\n"
        "}\n"
    )
    add_load_path(d)
    return d


def test_qualified_alias_resolves_from_dependency():
    _write_dep_fixture()
    src = """
    system "x" {
      dependency "dep-map" as remo
      stage main {
        sub remo.from_name remo.to_name
      }
    }
    """
    tree = isc_to_tree(src)
    subst = tree["stages"][0]["children"][0]
    assert subst["pattern"] == "404"
    assert subst["result"] == "500"


def test_imported_alias_resolves():
    _write_dep_fixture()
    src = """
    system "x" {
      dependency "dep-map"
      stage main {
        sub from_name to_name
      }
    }
    """
    tree = isc_to_tree(src)
    subst = tree["stages"][0]["children"][0]
    assert subst["pattern"] == "404"
    assert subst["result"] == "500"


@pytest.mark.skipif(
    not Path(os.environ.get("INTERSCRIPT_MAPS_PATH", Path(__file__).parent.parent.parent / "maps" / "maps")).is_dir(),
    reason="interscript maps repo not present",
)
def test_library_aliases_resolve_intra_library_references():
    """var-kor defines jamo in terms of other library aliases; the
    harvested value must have those substituted (they were previously
    dropped silently by the any-list tokenizer)."""
    from interscript.isc import _library_aliases

    libs = Path(os.environ.get("INTERSCRIPT_MAPS_PATH", Path(__file__).parent.parent.parent / "maps" / "maps")).parent / "libs"
    al = _library_aliases("var-kor", libs)
    assert "jamo_leading_cons" not in al["jamo"]
    assert "\\u1100" in al["jamo"] or "ᄀ" in al["jamo"]


# ---------------------------------------------------------------------------
# WO20 differential fuzz regressions — interscript-py#43 (parse hang) and
# interscript-py#44 (raw ValueError on malformed \u escapes).
# ---------------------------------------------------------------------------


def _parse_only(source: str, filename: str | None = None) -> dict:
    """The parse layer only. Both fuzz issues live in the parser; running
    conversion would conflate them with the engine's documented
    UnsupportedConstruct vocabulary errors."""
    from interscript.isc import _IscParser

    return _IscParser(source, filename).parse()


# -- interscript-py#43: truncated maps must terminate with IscParseError --


def test_truncated_stage_unterminated_sub_raises():
    """The campaign's 10-line repro: EOF inside `stage main` after a
    complete `sub { }` block, with the stage and system braces
    unterminated. Pre-fix this spun in _stage's skip-line fallback
    forever (the loop broke only on `}`)."""
    src = (
        'system "VAR:hin-Deva:Latn:hunterian-1872" {\n'
        "metadata {\n"
        "  notes {\n"
        "  }\n"
        "}\n"
        "tests {\n"
        "}\n"
        "stage main {\n"
        "  sub {\n"
        "  }\n"
    )
    with pytest.raises(IscParseError):
        _parse_only(src, "hang.isc")


def test_truncated_sequence_rule_block_raises():
    """Same class, inner loop: EOF inside a sequence/parallel rule block
    after a complete rule — the inner loop's skip-line fallback had the
    identical missing-EOF guard."""
    with pytest.raises(IscParseError):
        _parse_only('system "x" { stage main { sequence { sub "a" "b"')


def test_truncated_map_prefixes_terminate_named():
    """Class closure: every truncation of a real corpus map terminates
    with a document or a named IscParseError — never a spin, never an
    accidental ValueError/TypeError. The fixtures are real corpus maps,
    committed verbatim (var-ara-Arab-Arab-rababa, royin-tha-Thai-Latn-
    1999; dependency-free so the full text parses)."""
    fixtures = {
        "var-ara-Arab-Arab-rababa.isc": _RABABA_MAP,
        "royin-tha-Thai-Latn-1999.isc": _ROYIN_THA_MAP,
    }
    for name, text in fixtures.items():
        # The full map parses (the sweep is meaningless otherwise).
        _parse_only(text, name)
        for cut in range(len(text)):
            try:
                _parse_only(text[:cut], name)
            except IscParseError:
                continue


_RABABA_MAP = '''system "VAR:ara-Arab:Arab:rababa" {

metadata {
  authority_id var
  id rababa
  language iso-639-2:ara
  source_script Arab
  destination_script Arab
  name RABABA the Arabic Diacritization Library
  url https://github.com/interscript/rababa/
  description {    Arabic diacritization is useful for several practical business cases
    like text to speech or Romanization of Arabic texts or scripts.

    This map in particular uses the RABABA library to automatically detect
    and add diacritics using machine learning.
  }
}

tests {
   'قطر'  -> 'قِطْرَ'
   'abc' ->  'abc'
   '‘Iz. Ibrāhīm as-Sa‘danī' ->  '‘Iz. Ibrāhīm as-Sa‘danī'
}

stage main {
  rababa config: "200"
}
}
'''

_ROYIN_THA_MAP = '''system "ROYIN:tha-Thai:Latn:1999" {

metadata {
  authority_id royin
  id 1999
  language iso-639-2:tha
  source_script Thai
  destination_script Latn
  name Royal Thai General System of Transcription (1999) — phoneme-to-RTGS stage
  url http://www.royin.go.th/wp-content/uploads/royin-ebook/276/FileUpload/758_6484.pdf
  creation_date 1999
  adoption_date 2002
  description

  notes {
    "This is the PHONEME-to-RTGS tail stage of the Thai composite: it consumes a syllable-dotted phonemic reading (the output of a trained Thai reading layer, e.g. sequitur/ByT5 over the pythainlp thai-g2p corpus or the tha-phonemic lexicon) and renders the Royal Thai General System of Transcription.

    Historical: the v1 maps corpus carried this mapping as the post-rules of a seq2seq-dependent map (transcription: sequitur.pythainlp_lexicon); revived here as a standalone stage for the B6 composite (WO67).

    Tones (˨˩˦˧˥) and the syllable separator (.) are not romanized (the RTGS drops tone; the separator marks syllable bounds)."
  }
}

tests {
   "k a ˨˩ . k aː ˧" ->  "kaka"
   "k o ˨˩ . h a n ˩˩˦" ->  "kohan"
   "k o k̚ ˨˩ . t͡ɕʰ aː ŋ ˦˥" ->  "kokchang"
   "r a ˦˥ . d u ŋ ˧" ->  "radung"
}


stage main {

  # RTGS POST-RULES (revived from maps-staging/royin-tha-Thai-Latn-1999,
  # the seq2seq composite tail; the phoneme inventory is the pythainlp
  # thai-g2p / royalin one)
  parallel {
    sub "t͡ɕʰ" "ch"
    sub "t͡ɕ" "ch"
    sub "ʰ" "h"
    sub "ŋ" "ng"
    sub "j" "i"
    sub "w" "w"
    sub "ɔ" "o"
    sub "ɤ" "oe"
    sub "ɛ" "ae"
    sub "ɯ" "ue"
    sub "ʔ" ""
    sub "̚" ""
    sub "ː" ""
    sub "˨" ""
    sub "˩" ""
    sub "˦" ""
    sub "˧" ""
    sub "˥" ""
    sub "." ""
    sub " " ""
  }
}
}
'''


# -- interscript-py#44: malformed \u escapes name the error --


def test_malformed_escape_issue_repro_raises_named():
    """The issue's 90-byte repro: the braced form `\\u{1F600}` is not a
    grammar escape (esc_uni4 wants exactly 4 hex). Pre-fix this leaked a
    raw ValueError from int()."""
    src = (
        'system "T:fz-em6:1" {\n'
        "  stage main {\n"
        "    sequence {\n"
        '      sub "\\u{1F600}" "X"\n'
        "    }\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(IscParseError):
        isc_to_tree(src, "escape.isc")


def test_malformed_escape_matrix():
    """Every malformed lowercase \\u form is an IscParseError with
    position info; note text is scanned too (the issue's note-text
    finding); valid escapes still decode."""
    def expect_named_error(src: str) -> None:
        with pytest.raises(IscParseError) as excinfo:
            _parse_only(src)
        # Position pointer present (the IscParseError renders line:col).
        assert re.search(r"line \d+:\d+: ", str(excinfo.value))

    system = 'system "x" { stage main { sub %s "X" } }'
    # short hex (3 digits, then the closing quote)
    expect_named_error(system % '"\\u00E"')
    # non-hex head and tail
    expect_named_error(system % '"\\uZZZZ"')
    expect_named_error(system % '"\\u12G4"')
    # braced form
    expect_named_error(system % '"\\u{1F600}"')
    # escape cut by end-of-string / end-of-file
    expect_named_error(system % '"\\u00"')
    expect_named_error('system "x" { stage main { sub "\\u')

    # note text: the escape scanner runs over free note strings too
    expect_named_error(
        'system "x" {\nmetadata {\n  notes {\n    note "\\uZZZZ"\n  }\n}\n}'
    )


def test_valid_escape_still_decodes():
    """The fix is error-path only: a well-formed \\uXXXX decodes exactly
    as before."""
    doc = _parse_only('system "x" { stage main { sub "\\u00E9" "X" } }')
    rule = doc["stages"][0]["body"][0]["rule"]
    assert rule["from"] == {"type": "string", "value": "é"}


def test_capital_escape_stays_literal_u():
    """Capital \\U is NOT decoded by this parser: the grammar's esc_u_lone
    branch renders a literal "U" and the digits stay raw text (the ts
    hand-parser parity deviation in actions.ts; ruby's artifact parser
    decodes both spellings — cross-runtime alignment is an owner
    decision). Malformed \\U therefore never raises; it must also never
    crash (#44's adjacent shape)."""
    system = 'system "x" { stage main { sub %s "X" } }'
    doc = _parse_only(system % '"\\UZZZZ"')
    rule = doc["stages"][0]["body"][0]["rule"]
    assert rule["from"] == {"type": "string", "value": "UZZZZ"}
    doc = _parse_only(system % '"\\U00E"')
    rule = doc["stages"][0]["body"][0]["rule"]
    assert rule["from"] == {"type": "string", "value": "U00E"}
