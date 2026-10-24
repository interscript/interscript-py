import pytest

from interscript.engine import Engine
from interscript.equivalence import check
from interscript.isc import isc_to_tree

A = (
    'system "TSTA:xx-Latn:Latn:1" {\n'
    '  stage main { parallel {\n    sub "a" "x"\n  } }\n'
    "}\n"
)
B = (
    'system "TSTB:xx-Latn:Latn:2" {\n'
    '  stage main { parallel {\n    sub "a" "y"\n  } }\n'
    "}\n"
)
B_EQUIV = (
    'system "TSTB2:xx-Latn:Latn:3" {\n'
    '  stage main { parallel {\n    sub "a" "x"\n  } }\n'
    "}\n"
)


def test_equivalent_systems_report_ratio_one():
    r = check(A, B_EQUIV, ["ba", "a", "zzz"])
    assert r["tested"] == 3
    assert r["matched"] == 3
    assert r["ratio"] == 1.0
    assert r["distinguishing"] == []


def test_distinguishing_samples_are_collected():
    r = check(A, B, ["zzz", "ba"])
    assert r["tested"] == 2
    assert r["matched"] == 1
    assert r["ratio"] == 0.5
    assert r["distinguishing"] == [
        {"sample": "ba", "first": "bx", "second": "by"}
    ]


def test_empty_sample_set_is_vacuously_equivalent():
    assert check(A, B, [])["ratio"] == 1.0


@pytest.mark.parametrize("src", [A, B, B_EQUIV])
def test_sources_parse(src):
    Engine(isc_to_tree(src))
