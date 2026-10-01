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
    r"|(?P<space>\bspace\b)|(?P<boundary>\bboundary\b)"
    r"|(?P<nwb>\bnon_word_boundary\b)"
    r'|capture\(\s*(?P<grp>(?:[^()\\]|\\.|\([^()]*\))*)\s*\)' 
    r"|(?P<line_end>\bline_end\b)|(?P<line_start>\bline_start\b)"
    r"|(?P<cat>\+)"
)
def _split_list(src: str) -> list[str]:
    """Alternatives of any([...]) are full expressions — split on
    commas outside quotes and brackets (boundary + "ab" keeps its
    boundary atom; any([a, b]) nests)."""
    parts, buf, quote, depth = [], [], None, 0
    for ch in src:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            buf.append(ch)
            quote = ch
        elif ch == "[":
            depth += 1
            buf.append(ch)
        elif ch == "]":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _read_bracketed(expr: str, pos: int) -> tuple[str, int]:
    """With pos at the '[' of any([: return the inner content and the
    position after the matching ']' (quote- and depth-aware)."""
    depth, i, n = 0, pos, len(expr)
    while i < n:
        c = expr[i]
        if c == '"':
            i += 1
            while i < n:
                if expr[i] == "\\":
                    i += 2
                    continue
                if expr[i] == '"':
                    break
                i += 1
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return expr[pos + 1 : i], i + 1
        i += 1
    raise ValueError(f"unterminated any([ in {expr!r}")
_UNESC = re.compile(r"\\u([0-9a-fA-F]{4})")

SPACE = re.escape(" ")

# Ruby's \b counts combining marks as word characters; Python's \w
# does not (Mn is not alphanumeric). At a hamza-carrier + kasra
# junction Ruby sees no boundary while Python does — word-final rules
# fired wrongly and doubled vowels. Express the boundary as an
# explicit word/non-word transition over a word class that includes
# combining marks (Mnemonic ranges: combining diacritics 0300-036F,
# Arabic diacritics 064B-065F, 0670, and Quranic annotation 06D6-06ED).
_WORD = r"[\w\u0300-\u036F\u064B-\u065F\u0670\u06D6-\u06ED]"
_BOUNDARY = "(?:(?<=" + _WORD + ")(?!" + _WORD + ")|(?<!" + _WORD + ")(?=" + _WORD + "))"


def _unesc(s: str) -> str:
    return _UNESC.sub(lambda m: chr(int(m.group(1), 16)), s)


_ANY_LIST = re.compile(r"any\(\s*\[")


def _scan(expr: str, want: str):
    """Tokenize an expression into (kind, value); raises on gaps."""
    out: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        if expr[pos].isspace():
            pos += 1
            continue
        if _ANY_LIST.match(expr, pos):
            bracket = expr.find("[", pos)
            inner, pos = _read_bracketed(expr, bracket)
            while pos < len(expr) and expr[pos].isspace():
                pos += 1
            if pos >= len(expr) or expr[pos] != ")":
                raise ValueError(f"expected ')' closing any([ in {expr!r}")
            pos += 1
            out.append(("alt", "\x00".join(_split_list(inner))))
            continue
        m = _TOKEN.match(expr, pos)
        if not m:
            raise ValueError(f"cannot parse {want} near {expr[pos : pos + 20]!r} in {expr!r}")
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
        elif g["space"] is not None:
            out.append(("space", " "))
        elif g["boundary"] is not None:
            out.append(("boundary", ""))
        elif g["nwb"] is not None:
            out.append(("nwb", ""))
        elif g["grp"] is not None:
            out.append(("grp", g["grp"]))
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


def expr_neg_lookbehind(expr: str) -> str:
    """Negative lookbehind over an expression. Python re requires
    fixed-width lookbehinds (Ruby's Onigmo does not), so a top-level
    alternation is distributed: (?<!A|B) == (?<!A)(?<!B)."""
    toks = _scan(expr, "expression")
    if len(toks) == 1 and toks[0][0] == "alt":
        parts = [expr_to_regex(a) for a in toks[0][1].split("\x00")]
    else:
        parts = [expr_to_regex(expr)]
    return "".join(f"(?<!{p})" for p in parts)


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
            parts.append("(?:" + "|".join(expr_to_regex(a) for a in alts) + ")")
        elif kind == "opt":
            parts.append("(?:" + re.escape(value) + ")?")
        elif kind == "space":
            parts.append(SPACE)
        elif kind == "boundary":
            parts.append(_BOUNDARY)
        elif kind == "nwb":
            parts.append(r"\B")
        elif kind == "grp":
            parts.append("(" + expr_to_regex(value) + ")")
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
            parts.append(expr_to_literal(value.split("\x00")[0]))
        elif kind == "space":
            parts.append(" ")
        elif kind in ("boundary", "anchor"):
            raise ValueError(f"{kind} is not valid in a result expression")
    return "".join(parts)


def is_plain_string(expr: str) -> bool:
    return bool(re.fullmatch(r'"(?:[^"\\]|\\.)*"', expr.strip()))


def expr_max_length(expr: str) -> int:
    """The Ruby runtime's parallel-selection key: Rule::Sub#max_length =
    from + before + after + not_before + not_after (+ priority), where a
    zero-width stdlib alias (boundary, line_start, ...) counts 1."""
    total = 0
    for kind, value in _scan(expr, "expression"):
        if kind == "lit":
            total += len(value)
        elif kind in ("cls", "range", "space", "boundary", "nwb", "anchor"):
            total += 1
        elif kind == "alt":
            total += max(expr_max_length(a) for a in value.split("\x00"))
        elif kind == "opt":
            total += len(value)
        elif kind == "grp":
            total += expr_max_length(value)
        # cat: concatenation marker
    return total
