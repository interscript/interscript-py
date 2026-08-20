"""Expression layer for the .imp sub language.

Grammar (whitespace-insensitive, concatenated with +):
    expr := term ('+' term)*
    term := "literal" | any("chars") | any(["lit", ...]) | space | boundary
any("...") is a character class; any([...]) is an alternation of
literals. Compiles to Python regex fragments; results compile to plain
strings (any("ie") in a result takes the first alternative).
"""

from __future__ import annotations

import re

_TOKEN = re.compile(
    r'"(?P<lit>(?:[^"\\]|\\.)*)"'
    r'|any\(\s*"(?P<rlo>(?:[^"\\]|\\.)*)"\s*\.\.\s*"(?P<rhi>(?:[^"\\]|\\.)*)"\s*\)'
    r'|any\(\s*"(?P<cls>(?:[^"\\]|\\.)*)"\s*\)'
    r'|maybe\(\s*"(?P<opt>(?:[^"\\]|\\.)*)"\s*\)'
    r'|any\(\s*\[(?P<lst>(?:[^\\\[\]]|\\.)*)\]\s*\)'
    r"|(?P<space>\bspace\b)|(?P<boundary>\bboundary\b)"
    r"|(?P<line_end>\bline_end\b)|(?P<line_start>\bline_start\b)"
    r"|(?P<cat>\+)"
)
_LIST_SPLIT = re.compile(r'"((?:[^"\\]|\\.)*)"')
_UNESC = re.compile(r"\\u([0-9a-fA-F]{4})")

SPACE = re.escape(" ")


def _unesc(s: str) -> str:
    return _UNESC.sub(lambda m: chr(int(m.group(1), 16)), s)


def _scan(expr: str, want: str):
    """Tokenize an expression into (kind, value); raises on gaps."""
    out: list[tuple[str, str]] = []
    pos = 0
    for m in _TOKEN.finditer(expr):
        gap = expr[pos : m.start()]
        if gap.strip():
            raise ValueError(f"cannot parse {want} near {gap.strip()!r} in {expr!r}")
        pos = m.end()
        g = m.groupdict()
        if g["lit"] is not None:
            out.append(("lit", _unesc(g["lit"])))
        elif g["rlo"] is not None:
            out.append(("range", _unesc(g["rlo"]) + "\x00" + _unesc(g["rhi"])))
        elif g["cls"] is not None:
            out.append(("cls", _unesc(g["cls"])))
        elif g["opt"] is not None:
            out.append(("opt", _unesc(g["opt"])))
        elif g["lst"] is not None:
            alts = [_unesc(x) for x in _LIST_SPLIT.findall(g["lst"])]
            out.append(("alt", "\x00".join(alts)))
        elif g["space"] is not None:
            out.append(("space", " "))
        elif g["boundary"] is not None:
            out.append(("boundary", ""))
        elif g["line_end"] is not None:
            out.append(("anchor", "$"))
        elif g["line_start"] is not None:
            out.append(("anchor", "^"))
        else:
            out.append(("cat", ""))
    tail = expr[pos:]
    if tail.strip():
        raise ValueError(f"cannot parse {want} near {tail.strip()!r} in {expr!r}")
    if not out:
        raise ValueError(f"empty expression {expr!r}")
    return out


def expr_to_regex(expr: str) -> str:
    parts = []
    for kind, value in _scan(expr, "expression"):
        if kind == "lit":
            parts.append(re.escape(value))
        elif kind == "cls":
            parts.append("[" + re.escape(value) + "]")
        elif kind == "range":
            lo, hi = value.split("\x00")
            parts.append("[" + re.escape(lo) + "-" + re.escape(hi) + "]")
        elif kind == "alt":
            alts = value.split("\x00")
            parts.append("(?:" + "|".join(re.escape(a) for a in alts) + ")")
        elif kind == "opt":
            parts.append("(?:" + re.escape(value) + ")?")
        elif kind == "space":
            parts.append(SPACE)
        elif kind == "boundary":
            parts.append(r"\b")
        elif kind == "anchor":
            parts.append(value)
    return "".join(parts)


def expr_to_literal(expr: str) -> str:
    parts = []
    for kind, value in _scan(expr, "result"):
        if kind == "lit":
            parts.append(value)
        elif kind == "cls":
            parts.append(value[0])  # deterministic: first alternative
        elif kind == "alt":
            parts.append(value.split("\x00")[0])
        elif kind == "space":
            parts.append(" ")
        elif kind in ("boundary", "anchor"):
            raise ValueError(f"{kind} is not valid in a result expression")
    return "".join(parts)


def is_plain_string(expr: str) -> bool:
    return bool(re.fullmatch(r'"(?:[^"\\]|\\.)*"', expr.strip()))
