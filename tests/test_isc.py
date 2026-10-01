"""Unit tests for the ISC parser and converter (interscript.isc)."""

from __future__ import annotations

import os
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
