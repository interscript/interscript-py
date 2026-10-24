"""A3 (WO59): mechanical equivalence checking between two systems.

Mirrors the ruby Interscript::Equivalence contract: probe a sample set
through both systems, report the agreement ratio plus the distinguishing
samples.
"""

from __future__ import annotations

from .engine import Engine
from .isc import isc_to_tree


def check(first_source: str, second_source: str, samples: list[str]) -> dict:
    first = Engine(isc_to_tree(first_source))
    second = Engine(isc_to_tree(second_source))
    distinguishing = []
    tested = 0
    matched = 0
    for sample in samples:
        tested += 1
        a = first.transliterate(sample)
        b = second.transliterate(sample)
        if a == b:
            matched += 1
        else:
            distinguishing.append({"sample": sample, "first": a, "second": b})
    ratio = 1.0 if tested == 0 else matched / tested
    return {
        "tested": tested,
        "matched": matched,
        "ratio": ratio,
        "distinguishing": distinguishing,
    }
