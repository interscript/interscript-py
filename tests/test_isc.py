"""Unit tests for the ISC parser and converter (interscript.isc)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from interscript.isc import IscParseError, UnsupportedConstruct, isc_to_tree


SAMPLE = r'''
system "test-map-Cyrl-Latn-2000" {

metadata {
  name "Test Map"
  description {
    A description with \{ nested \} braces.
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

    # note text: the escape scanner runs over free note strings too —
    # BUT the reference grammar's ordered choice gives the notes block a
    # raw-text fallback (the note-line production fails on the malformed
    # string, the generic-field production re-reads the whole block as
    # prose; ruby: metadata notes = Slice 'note "\uZZZZ"\n  '). Only
    # RULE strings reject malformed escapes.
    doc = _parse_only(
        'system "x" {\nmetadata {\n  notes {\n    note "\\uZZZZ"\n  }\n}\n}'
    )
    assert doc["metadata"]["notes"] == 'note "\\uZZZZ"'


def test_valid_escape_still_decodes():
    """The fix is error-path only: a well-formed \\uXXXX decodes exactly
    as before."""
    doc = _parse_only('system "x" { stage main { sub "\\u00E9" "X" } }')
    rule = doc["stages"][0]["body"][0]["rule"]
    assert rule["from"] == {"type": "string", "value": "é"}


def test_capital_escape_decodes_and_falls_back():
    """Capital \\U follows the grammar's ordered alternatives: esc_uni
    decodes 4..8 hex digits (greedy) to one code point; with fewer than
    4 hex digits esc_u_lone renders a literal "U" and the digits stay
    raw text. Both spellings decode — the WO19 G6 ruling (ruby/rs
    always decoded; ts since the gki-bel "\\U040E" fix).

    The capital form was previously rendered as a bare "U" with the
    digits as raw text (a stale 0.1.x ts hand-parser parity note), so
    the three Belarusian uppercase-Ў rows (gki-bel-2000 / mvd-bel-2008
    / odni-bel-2015, `sub "\\U040E" "W"-style`) never fired and
    diverged from the rs golden corpus.

    Edge shapes are ruby-measured (4/4 with this parser):
    "\\U00000041" → "A"; "\\U000000410" → "A0" (first 8 hex consumed,
    the 9th stays raw); "\\U00e9" → "é"; "\\Uab" → "Uab". Malformed
    capital forms never raise (#44's adjacent shape).
    """
    system = 'system "x" { stage main { sub %s "X" } }'

    def from_value(literal):
        doc = _parse_only(system % literal)
        return doc["stages"][0]["body"][0]["rule"]["from"]["value"]

    # esc_uni: 4..8 hex digits decode, both hex casings
    assert from_value('"\\U040E"') == "Ў"
    assert from_value('"\\U045E"') == "ў"
    assert from_value('"\\U00e9"') == "é"
    assert from_value('"\\U00000041"') == "A"
    # greedy 8-hex cap: a 9th hex digit stays raw text
    assert from_value('"\\U000000410"') == "A0"
    # esc_u_lone fallback: fewer than 4 hex digits — literal "U",
    # digits raw, and never a raise
    assert from_value('"\\Uab"') == "Uab"
    assert from_value('"\\U00E"') == "U00E"
    assert from_value('"\\UZZZZ"') == "UZZZZ"


# --- maps#293 (WO20 fuzz): the parse accept/reject boundary must match
# --- the reference trio (ruby/ts/rs parse the vendored isc.parg grammar).
# --- Each family below is one grammar-strictness class, not a case fix.


def test_nested_block_inside_sequence_rejected():
    """A block keyword in rule position is a parse error. Grammar:
    rule_line = rule / use_directive / comment_item — sequence/parallel
    do not re-enter. The parity audit pins this as a negative probe
    rejected 4/4; py used to absorb the nested block's rules into the
    outer sequence (silently different rule set)."""
    src = '''
system "T:fz-em7:1" {
  stage main {
    sequence {
      sub "a" "b"
      parallel {
        sub "c" "d"
      }
    }
  }
}
'''
    with pytest.raises(IscParseError):
        isc_to_tree(src)


def test_unknown_keyword_in_stage_position_rejected():
    """An unrecognized keyword in stage-item position is a parse error
    (grammar stage_item); py used to silently drop the line."""
    src = '''
system "T:fz-ukw:1" {
  stage main {
    sequence {
      sub "a" "b"
      frobnicate "c" "d"
    }
  }
}
'''
    with pytest.raises(IscParseError):
        isc_to_tree(src)


def test_unknown_directive_in_stage_position_rejected():
    """Same class, block-less form: an unknown top-of-stage keyword."""
    src = '''
system "T:fz-ukw2:1" {
  stage main {
    frobnicate "a" "b"
  }
}
'''
    with pytest.raises(IscParseError):
        isc_to_tree(src)


def test_malformed_simple_escape_rejected():
    """The grammar's escape set is n r t " \\ uXXXX (plus the \\U forms);
    any other letter after the backslash fails the string production.
    py used to pass unknown escapes through as the bare letter."""
    with pytest.raises(IscParseError):
        isc_to_tree('system "x" { stage main { sub "a\\q" "b" } }')
    with pytest.raises(IscParseError):
        isc_to_tree('system "x" { stage main { sub "a\\z" "b" } }')


def test_valid_simple_escapes_still_decode():
    """The rejection is error-path only: the named escapes decode."""
    doc = _parse_only('system "x" { stage main { sub "a\\n\\t\\r\\"\\\\" "b" } }')
    rule = doc["stages"][0]["body"][0]["rule"]
    assert rule["from"] == {"type": "string", "value": 'a\n\t\r"\\'}


def test_ruleset_inside_metadata_rejected():
    """metadata_field has no ruleset production; the trio reject. py's
    generic-field fallback used to swallow the line (braces included)
    as a string value."""
    src = '''
system "T:fz-rim:1" {
  metadata {
    authority_id fz
    name "ruleset inside metadata"
    ruleset inner { sub "a" "b" }
  }
  stage main {
    sequence {
      sub "a" "b"
    }
  }
}
'''
    with pytest.raises(IscParseError):
        isc_to_tree(src)


def test_zero_width_primitive_in_compound_from_parses_then_names_at_exec():
    """A zero-width primitive inside a compound from is a GRAMMAR-legal
    document: the trio parse it and name the primitive at execution
    (ruby/ts 'Alias word_boundary not found', rs 'unknown primitive').
    py used to reject at conversion — the failure moves to the engine,
    named, under the existing unsupported-op path."""
    from interscript.engine import Engine, ExecutionError

    src = '''
system "FZ:grammar:350509" {
  stage main {
    sequence {
      sub {
        from word_boundary + capture(some("z")) + word_boundary
        to "W" + ref(1)
        not_before "j"
        not_after "k"
      }
    }
  }
}
'''
    tree = isc_to_tree(src)  # parse leg: ACCEPTS
    engine = Engine(tree)
    with pytest.raises(ExecutionError, match="word_boundary"):
        engine.transliterate("zz")  # exec leg: named error


# --- maps#295 (WO20 fuzz): metadata free-text scanning must be inert.


def test_note_text_brace_is_inert():
    """A bare '{' inside description free text is ORDINARY TEXT (grammar
    raw_text = *( "\\"{" / "\\"}" / (!"}" chr) ) — no brace-depth
    tracking). Brace-depth tracking made one '{' swallow the rest of the
    document (notes, tests, the stage block) into a metadata string, so
    the parsed rule set — and the transliteration output — silently
    changed with a one-character prose edit."""
    clean = '''
system "DIN:kaa-Cyrl:Latn:1460-2-2011" {
  metadata {
    name "DIN 1460-2:2011-10, Tabelle 1 row 22"
    description {
      Romanization of Cyrillic alphabets; extracted from the
      standard's text layer.
    }
    notes {
      note "Extraction provenance: the PDF text layer."
    }
  }
  tests {
     "a" ->  "a"
     "b" ->  "c"
  }
  stage main {
    parallel {
      sub "a" "a"
      sub "b" "c"
    }
  }
}
'''
    # one character inside the description prose flips: standard's -> standa{d's
    mutated = clean.replace("standard's", "standa{d's")
    clean_tree = isc_to_tree(clean)
    mutated_tree = isc_to_tree(mutated)
    assert [s["name"] for s in clean_tree["stages"]] == ["main"]
    assert clean_tree["stages"] == mutated_tree["stages"]

    from interscript.engine import Engine

    assert Engine(clean_tree).transliterate("ab") == "ac"
    assert Engine(mutated_tree).transliterate("ab") == "ac"


# --- maps#294 (WO20 fuzz): the self-run repro joins the mutual-cycle
# --- pin; every runtime names the cycle, none crash.


def test_self_run_cycle_raises():
    src = '''
system "T:fz-selfrun:1" {
  stage main {
    parallel {
      sub "a" "b"
    }
    run stage.main
  }
}
'''
    with pytest.raises(UnsupportedConstruct, match="cyclic"):
        isc_to_tree(src)


def test_content_after_system_block_rejected():
    """The reference grammars anchor isc_source at end-of-input: content
    after the system block is a parse error. Without the anchor, a
    mis-scoped metadata block closed the system early and the trailing
    tests/stages silently vanished from the parsed tree (the trio parse
    the same document to a parse error)."""
    src = '''
system "T:fz-tail:1" {
  metadata {
    description {
      A description with { nested } braces.
    }
  }
  stage main {
    parallel {
      sub "a" "b"
    }
  }
}
'''
    with pytest.raises(IscParseError):
        isc_to_tree(src)


def test_escaped_braces_in_description_decode():
    """The legal spelling for prose braces: \\{ and \\} decode to literal
    braces inside description free text (grammar raw_text escape
    branches); the document parses and the stage survives."""
    src = '''
system "T:fz-esc:1" {
  metadata {
    description {
      Braces \\{ escaped \\} in prose.
    }
  }
  stage main {
    parallel {
      sub "a" "b"
    }
  }
}
'''
    tree = isc_to_tree(src)
    assert tree["metadata"]["description"] == "Braces { escaped } in prose."
    assert [s["name"] for s in tree["stages"]] == ["main"]
