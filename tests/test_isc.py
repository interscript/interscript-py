"""Unit tests for the ISC parser and converter (interscript.isc)."""

from __future__ import annotations

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
    assert stage_children[0] == {"kind": "run", "map": "other-map"}


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
    bad = 'system "x" { stage main { parallel { sub some("a") "b" } } }'
    with pytest.raises(UnsupportedConstruct, match="some"):
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
        'system "x" { stage main { sequence { sub some("c") "d" } } }',
        on_unsupported="skip",
    )
    assert tree2["skipped_unsupported"]
    assert tree2["stages"][0]["children"] == []
