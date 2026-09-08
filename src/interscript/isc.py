"""Parser and converter for the ISC map format.

Parses ``.isc`` source (the only format the maps corpus still ships)
into the plain-tree shape the existing engine consumes — the engine and
the ``.imp`` parser are untouched (OCP). The grammar mirrors the
reference TypeScript parser (``interscript-ts/src/isc/``), which
mirrors the Ruby Parslet grammar.

Unsupported constructs (captures in results, ``some``) render loudly as
``UnsupportedConstruct`` — the same contract the engine applies to
unsupported stage kinds. Adding support = extending ``_render_item``,
nothing else changes.

Example:
    tree = isc_to_tree(source, "bgnpcgn-ukr-Cyrl-Latn-2019.isc")
    engine = Engine(tree)
    engine.transliterate("Антон")
"""

from __future__ import annotations

import re
from pathlib import Path

from .expr import expr_to_regex


class IscParseError(ValueError):
    """Raised when .isc source cannot be parsed."""

    def __init__(self, message: str, line: int, col: int, filename: str | None = None) -> None:
        prefix = f"{filename}: " if filename else ""
        super().__init__(f"{prefix}line {line}:{col}: {message}")


class UnsupportedConstruct(ValueError):
    """Raised when the ISC tree uses an item the engine vocabulary lacks."""


class _SplitParallel(Exception):
    """Internal: a parallel block with capture-bearing rules.

    Carries the capture-free subs and the capture-bearing rules so the
    converter can emit ordered substitutions ahead of the parallel op.
    """

    def __init__(self, subs: list[dict], capture_rules: list[dict]) -> None:
        super().__init__()
        self.subs = subs
        self.capture_rules = capture_rules


PRIMITIVES = {"boundary", "line_start", "line_end", "word_boundary", "non_boundary", "space"}
_FUNCTIONS = {"upcase", "downcase", "title_case", "reverse", "strip", "swapcase"}
_CONSTRAINTS = {"before", "after", "not_before", "not_after"}
# Tokens that terminate an item inside a rule; a bare word equal to one
# of these is a keyword, never an alias reference.
_KEYWORDS = _CONSTRAINTS | {"to", "from", "note"}

class _IscParser:
    """Recursive-descent ISC parser; one method per construct."""

    def __init__(self, src: str, filename: str | None) -> None:
        self._src = src
        self._filename = filename
        self._pos = 0

    # -- entry --

    def parse(self) -> dict:
        self._skip_ws()
        self._expect("system")
        self._skip_ws()
        system_code = self._string_literal()
        self._skip_ws()
        self._expect("{")
        body = self._block_items()
        self._skip_ws()
        self._expect("}")

        doc: dict = {
            "systemCode": system_code,
            "metadata": {},
            "tests": [],
            "aliases": [],
            "stages": [],
            "dependencies": [],
        }
        for item in body:
            if "metadata" in item:
                doc["metadata"].update(item["metadata"])
            elif "tests" in item:
                doc["tests"] = item["tests"]
            elif "aliases" in item:
                doc["aliases"] = item["aliases"]
            elif "stage" in item:
                doc["stages"].append(item["stage"])
            elif "dependency" in item:
                doc["dependencies"].append(item["dependency"])
        return doc

    # -- block items --

    def _block_items(self) -> list[dict]:
        items: list[dict] = []
        while True:
            self._skip_ws()
            if self._peek() == "}" or self._pos >= len(self._src):
                return items
            keyword = self._peek_word()
            if keyword == "metadata":
                items.append(self._metadata())
            elif keyword == "tests":
                items.append(self._tests())
            elif keyword == "aliases":
                items.append(self._aliases())
            elif keyword == "stage":
                items.append(self._stage())
            elif keyword == "dependency":
                items.append(self._dependency())
            else:
                self._error(f"Unexpected keyword: {keyword}")

    def _metadata(self) -> dict:
        self._expect("metadata")
        self._skip_ws()
        self._expect("{")
        metadata: dict = {}
        while True:
            self._skip_ws()
            if self._peek() == "}":
                break
            word = self._peek_word()
            if word == "description":
                self._consume("description")
                self._skip_inline_ws()
                if self._peek() == "{":
                    self._expect("{")
                    metadata["description"] = self._raw_text()
                    self._expect("}")
                elif self._peek() == '"':
                    metadata["description"] = self._string_literal()
                else:
                    metadata["description"] = self._to_newline().strip()
                continue
            if word == "notes":
                self._consume("notes")
                self._skip_inline_ws()
                if self._peek() == "{":
                    self._expect("{")
                    saved = self._pos
                    self._skip_ws()
                    is_note_list = self._peek_word() == "note"
                    self._pos = saved
                    if is_note_list:
                        notes = []
                        while True:
                            self._skip_ws()
                            if self._peek() == "}":
                                break
                            self._expect("note")
                            self._skip_ws()
                            notes.append(self._string_literal())
                        self._expect("}")
                        metadata["notes"] = notes
                    else:
                        metadata["notes"] = self._raw_text()
                        self._expect("}")
                continue
            key = self._identifier()
            self._skip_inline_ws()
            if self._peek() in ("\n", "}") and self._peek() != "{":
                metadata[key] = ""
                if self._peek() != "}":
                    continue
                continue
            if self._peek() == "{":
                self._expect("{")
                metadata[key] = self._raw_text()
                self._expect("}")
            elif self._peek() == '"':
                metadata[key] = self._string_literal()
            else:
                metadata[key] = self._to_newline().strip()
        self._expect("}")
        return {"metadata": metadata}

    def _tests(self) -> dict:
        self._expect("tests")
        self._skip_ws()
        self._expect("{")
        tests = []
        while True:
            self._skip_ws()
            if self._peek() == "}":
                break
            if self._peek() == "#":
                self._skip_line()
                continue
            input_text = self._string_literal()
            self._skip_ws()
            self._expect("->")
            self._skip_ws()
            expected = self._string_literal()
            self._skip_ws()
            if self._peek_word() == "note":
                self._consume("note")
                self._skip_ws()
                note = self._string_literal()
                tests.append({"input": input_text, "expected": expected, "note": note})
            else:
                tests.append({"input": input_text, "expected": expected})
        self._expect("}")
        return {"tests": tests}

    def _aliases(self) -> dict:
        self._expect("aliases")
        self._skip_ws()
        self._expect("{")
        aliases = []
        while True:
            self._skip_ws()
            if self._peek() == "}":
                break
            if self._peek() == "#":
                self._skip_line()
                continue
            name = self._identifier()
            self._skip_ws()
            self._expect("=")
            self._skip_ws()
            aliases.append({"name": name, "value": self._item()})
        self._expect("}")
        return {"aliases": aliases}

    def _stage(self) -> dict:
        self._expect("stage")
        self._skip_ws()
        name = self._identifier()
        self._skip_ws()
        self._expect("{")
        body: list[dict] = []
        while True:
            self._skip_ws()
            if self._peek() == "}":
                break
            if self._peek() == "#":
                self._skip_line()
                continue
            keyword = self._peek_word()
            if keyword in ("parallel", "sequence"):
                self._consume(keyword)
                self._skip_ws()
                self._expect("{")
                rules = []
                while True:
                    self._skip_ws()
                    if self._peek() == "}":
                        break
                    if self._peek() == "#":
                        self._skip_line()
                        continue
                    if self._peek_word() == "sub":
                        rules.append(self._rule())
                    else:
                        # Ruby Parslet silently drops unrecognized tokens
                        # inside rule blocks; parity.
                        self._skip_line()
                self._expect("}")
                body.append({"kind": keyword, "rules": rules})
            elif keyword == "run":
                self._consume("run")
                self._skip_ws()
                dep = None
                if self._peek_word() == "map":
                    self._consume("map")
                    self._expect(".")
                    dep = self._identifier()
                    self._expect(".")
                    self._expect("stage")
                    self._expect(".")
                    stage_name = self._identifier()
                elif self._peek_word() == "stage":
                    self._consume("stage")
                    self._expect(".")
                    stage_name = self._identifier()
                else:
                    self._error("Expected map. or stage. after run")
                body.append({"kind": "run", "dependency": dep, "stage": stage_name})
            elif keyword == "separate":
                self._consume("separate")
                self._skip_ws()
                if self._peek_word() == "separator":
                    self._consume("separator")
                    self._skip_ws()
                    body.append({"kind": "separate", "separator": self._item_atom()})
                else:
                    body.append({"kind": "separate"})
            elif keyword in ("compose", "decompose"):
                self._consume(keyword)
                body.append({"kind": keyword})
            elif keyword == "rababa":
                self._consume("rababa")
                self._skip_inline_ws()
                kwargs = {}
                while self._peek() not in ("}", "", "\n"):
                    arg_name = self._identifier()
                    if not arg_name:
                        break
                    self._skip_inline_ws()
                    self._expect(":")
                    self._skip_inline_ws()
                    kwargs[arg_name] = self._string_literal()
                    self._skip_inline_ws()
                    if self._peek() == ",":
                        self._pos += 1
                    self._skip_inline_ws()
                body.append({"kind": "funcall", "name": "rababa", "kwargs": kwargs})
            elif keyword in _FUNCTIONS:
                self._consume(keyword)
                body.append({"kind": "string_case", "op": keyword})
            elif keyword == "sub":
                body.append({"kind": "bare_rule", "rule": self._rule()})
            else:
                # Ruby Parslet parity: drop unrecognized tokens.
                self._skip_line()
        self._expect("}")
        return {"stage": {"name": name, "body": body}}

    # -- rules --

    def _rule(self) -> dict:
        self._expect("sub")
        self._skip_ws()
        if self._peek() == "{":
            self._expect("{")
            rule = {"from": {"type": "none"}, "to": {"type": "none"}, "constraints": []}
            while True:
                self._skip_ws()
                if self._peek() == "}":
                    break
                word = self._peek_word()
                if word == "from":
                    self._consume("from")
                    self._skip_ws()
                    rule["from"] = self._item()
                elif word == "to":
                    self._consume("to")
                    self._skip_ws()
                    rule["to"] = self._item()
                elif word in _CONSTRAINTS:
                    self._consume(word)
                    self._skip_ws()
                    rule["constraints"].append({"kind": word, "item": self._item()})
                else:
                    self._error(f"Unexpected keyword in sub block: {word}")
            self._expect("}")
            return rule
        from_item = self._item_atom()
        self._skip_ws()
        to_item = self._item_atom()
        constraints = []
        while True:
            self._skip_ws()
            word = self._peek_word()
            if word not in _CONSTRAINTS:
                break
            self._consume(word)
            self._skip_ws()
            constraints.append({"kind": word, "item": self._item_atom()})
        return {"from": from_item, "to": to_item, "constraints": constraints}

    # -- items --

    def _item(self) -> dict:
        parts = [self._item_atom()]
        while True:
            self._skip_inline_ws()
            if self._peek() == "+":
                self._pos += 1
                self._skip_inline_ws()
                parts.append(self._item_atom())
            elif self._is_item_atom_start() and self._peek_word() not in _KEYWORDS:
                parts.append(self._item_atom())
            else:
                break
        if len(parts) == 1:
            return parts[0]
        return {"type": "concat", "parts": parts}

    def _item_atom(self) -> dict:
        c = self._peek()
        if c in ('"', "'"):
            return {"type": "string", "value": self._string_literal()}
        word = self._peek_word()
        if not word:
            self._error("Expected item atom")
        if word == "none":
            self._consume("none")
            return {"type": "none"}
        if word == "any_character":
            self._consume("any_character")
            return {"type": "function", "name": "any_character"}
        if word == "any":
            self._consume("any")
            self._expect("(")
            self._skip_ws()
            inner = self._peek()
            if inner in ('"', "'"):
                lo = self._string_literal()
                self._skip_ws()
                if self._src.startswith("..", self._pos):
                    self._pos += 2
                    self._skip_ws()
                    hi = self._string_literal()
                    self._skip_ws()
                    self._expect(")")
                    return {"type": "range", "lo": lo, "hi": hi}
                self._expect(")")
                return {
                    "type": "set",
                    "items": [{"type": "string", "value": ch} for ch in lo],
                }
            if inner == "[":
                self._expect("[")
                items = []
                while True:
                    self._skip_ws()
                    if self._peek() == "]":
                        break
                    items.append(self._item())
                    self._skip_ws()
                    if self._peek() == ",":
                        self._pos += 1
                self._expect("]")
                self._skip_ws()
                self._expect(")")
                return {"type": "set", "items": items}
            item = self._item()
            self._skip_ws()
            self._expect(")")
            return {"type": "set", "items": [item]}
        if word == "capture":
            self._consume("capture")
            self._expect("(")
            self._skip_ws()
            inner = self._item()
            self._skip_ws()
            self._expect(")")
            return {"type": "capture_group", "inner": inner}
        if word == "maybe":
            self._consume("maybe")
            self._expect("(")
            self._skip_ws()
            inner = self._item()
            self._skip_ws()
            self._expect(")")
            return {"type": "maybe", "inner": inner}
        if word == "some":
            self._consume("some")
            self._expect("(")
            self._skip_ws()
            inner = self._item()
            self._skip_ws()
            self._expect(")")
            return {"type": "some", "inner": inner}
        if word == "ref":
            self._consume("ref")
            self._expect("(")
            self._skip_ws()
            num = self._number()
            self._skip_ws()
            self._expect(")")
            return {"type": "capture", "index": num}
        if word in PRIMITIVES:
            self._consume(word)
            return {"type": "primitive", "name": word}
        if word in _FUNCTIONS:
            self._consume(word)
            return {"type": "function", "name": word}
        self._consume(word)
        return {"type": "alias_ref", "name": word}

    def _dependency(self) -> dict:
        self._expect("dependency")
        self._skip_ws()
        target = self._string_literal()
        self._skip_inline_ws()
        if self._peek_word() == "as":
            self._consume("as")
            self._skip_ws()
            return {"dependency": {"target": target, "aliasName": self._identifier()}}
        return {"dependency": {"target": target}}

    # -- primitives --

    def _string_literal(self) -> str:
        c = self._peek()
        if c == '"':
            return self._quoted_string()
        if c == "'":
            return self._single_quoted()
        self._error(f"Expected string but got {self._src[self._pos : self._pos + 20]!r}")

    def _quoted_string(self) -> str:
        self._expect('"')
        result = ""
        while self._pos < len(self._src):
            c = self._src[self._pos]
            self._pos += 1
            if c == '"':
                return result
            if c == "\\":
                nxt = self._src[self._pos] if self._pos < len(self._src) else ""
                self._pos += 1
                if nxt == "u":
                    hex4 = self._src[self._pos : self._pos + 4]
                    self._pos += 4
                    result += chr(int(hex4, 16))
                else:
                    result += {"n": "\n", "t": "\t", "r": "\r"}.get(nxt, nxt)
            else:
                result += c
        self._error("Unterminated string")

    def _single_quoted(self) -> str:
        self._expect("'")
        start = self._pos
        while self._pos < len(self._src) and self._src[self._pos] != "'":
            self._pos += 1
        result = self._src[start : self._pos]
        self._expect("'")
        return result

    def _raw_text(self) -> str:
        result = ""
        depth = 0
        while self._pos < len(self._src):
            c = self._src[self._pos]
            if c == "\\" and self._pos + 1 < len(self._src) and self._src[self._pos + 1] in "{}":
                result += self._src[self._pos + 1]
                self._pos += 2
                continue
            if c == "}" and depth == 0:
                break
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            result += c
            self._pos += 1
        return result.strip()

    def _identifier(self) -> str:
        self._skip_ws()
        start = self._pos
        while self._pos < len(self._src) and (self._src[self._pos].isalnum() or self._src[self._pos] in "_-"):
            self._pos += 1
        if self._pos == start:
            self._error("Expected identifier")
        return self._src[start : self._pos]

    def _number(self) -> int:
        start = self._pos
        while self._pos < len(self._src) and self._src[self._pos].isdigit():
            self._pos += 1
        return int(self._src[start : self._pos] or "0")

    def _to_newline(self) -> str:
        start = self._pos
        while self._pos < len(self._src) and self._src[self._pos] != "\n":
            self._pos += 1
        return self._src[start : self._pos]

    # -- helpers --

    def _peek(self) -> str:
        return self._src[self._pos] if self._pos < len(self._src) else ""

    def _peek_word(self) -> str:
        start = self._pos
        while self._pos < len(self._src) and (self._src[self._pos].isalnum() or self._src[self._pos] == "_"):
            self._pos += 1
        word = self._src[start : self._pos]
        self._pos = start
        return word

    def _expect(self, s: str) -> None:
        self._skip_ws()
        if not self._src.startswith(s, self._pos):
            self._error(f"Expected {s!r} but got {self._src[self._pos : self._pos + 20]!r}")
        self._pos += len(s)

    def _consume(self, word: str) -> None:
        if not self._src.startswith(word, self._pos):
            self._error(f"Expected {word!r}")
        self._pos += len(word)

    def _skip_ws(self) -> None:
        while self._pos < len(self._src):
            c = self._src[self._pos]
            if c.isspace():
                self._pos += 1
                continue
            if c == "#":
                self._skip_line()
                continue
            return

    def _skip_inline_ws(self) -> None:
        while self._pos < len(self._src) and self._src[self._pos] in " \t":
            self._pos += 1

    def _skip_line(self) -> None:
        while self._pos < len(self._src) and self._src[self._pos] != "\n":
            self._pos += 1
        if self._pos < len(self._src):
            self._pos += 1

    def _is_item_atom_start(self) -> bool:
        c = self._peek()
        return c in ('"', "'") or c.isalnum() or c == "_"

    def _error(self, message: str) -> None:
        line = self._src.count("\n", 0, self._pos) + 1
        col = self._pos - (self._src.rfind("\n", 0, self._pos) + 1) + 1
        raise IscParseError(message, line, col, self._filename)


# ---------------------------------------------------------------------------
# Converter: ISC document -> the engine's plain tree.
# ---------------------------------------------------------------------------


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _render_item(item: dict, aliases: dict[str, str]) -> str:
    """Render an IscItem as the expression source the engine compiles."""
    kind = item["type"]
    if kind == "string":
        return f'"{_escape(item["value"])}"'
    if kind == "none":
        return '""'
    if kind == "concat":
        return " + ".join(_render_item(part, aliases) for part in item["parts"])
    if kind == "set":
        items = item["items"]
        if all(sub["type"] == "string" for sub in items):
            return f'any("{_escape("".join(sub["value"] for sub in items))}")'
        return "any([" + ", ".join(_render_item(sub, aliases) for sub in items) + "])"
    if kind == "range":
        return f'any("{_escape(item["lo"])}".."{_escape(item["hi"])}")'
    if kind == "maybe":
        inner = item["inner"]
        if inner["type"] != "string":
            raise UnsupportedConstruct("maybe() over non-string items")
        return f'maybe("{_escape(inner["value"])}")'
    if kind == "primitive":
        name = item["name"]
        if name == "space":
            return "space"
        if name == "boundary":
            return "boundary"
        if name == "line_start":
            return "line_start"
        if name == "line_end":
            return "line_end"
        raise UnsupportedConstruct(f"primitive {name}")
    if kind == "alias_ref":
        name = item["name"]
        if name not in aliases:
            raise UnsupportedConstruct(f"unresolved alias {name}")
        return aliases[name]
    raise UnsupportedConstruct(
        f"item kind {kind} (captures/some/functions are not in the engine vocabulary)"
    )


def _has_capture(item: dict) -> bool:
    """True when an IscItem tree contains capture_group/capture nodes."""
    if not isinstance(item, dict):
        return False
    if item.get("type") in ("capture_group", "capture"):
        return True
    if any(_has_capture(sub) for sub in item.get("parts", [])):
        return True
    if any(_has_capture(sub) for sub in item.get("items", [])):
        return True
    if "inner" in item:
        return _has_capture(item["inner"])
    return False


def _regex_of(item: dict, aliases: dict[str, str]) -> str:
    """Render an IscItem as a regex fragment (subst-family rules)."""
    kind = item["type"]
    if kind == "string":
        return re.escape(item["value"])
    if kind == "none":
        return ""
    if kind == "concat":
        return "".join(_regex_of(part, aliases) for part in item["parts"])
    if kind == "set":
        items = item["items"]
        if all(sub["type"] == "string" for sub in items):
            return "[" + re.escape("".join(sub["value"] for sub in items)) + "]"
        return "(?:" + "|".join(_regex_of(sub, aliases) for sub in items) + ")"
    if kind == "range":
        return "[" + re.escape(item["lo"]) + "-" + re.escape(item["hi"]) + "]"
    if kind == "maybe":
        return "(?:" + _regex_of(item["inner"], aliases) + ")?"
    if kind == "capture_group":
        return "(" + _regex_of(item["inner"], aliases) + ")"
    if kind == "primitive":
        return {
            "boundary": r"\b",
            "line_start": "^",
            "line_end": "$",
            "space": " ",
        }.get(item["name"], None) or _unsupported_primitive(item["name"])
    if kind == "alias_ref":
        name = item["name"]
        if name not in aliases:
            raise UnsupportedConstruct(f"unresolved alias {name}")
        return aliases[name]
    raise UnsupportedConstruct(f"item kind {kind} in a pattern")


def _unsupported_primitive(name: str) -> str:
    raise UnsupportedConstruct(f"primitive {name}")


def _repl_of(item: dict, aliases: dict[str, str]) -> str:
    """Render an IscItem as a replacement string ($N backrefs allowed)."""
    kind = item["type"]
    if kind == "string":
        return item["value"].replace("\\", "\\\\")
    if kind == "none":
        return ""
    if kind == "concat":
        return "".join(_repl_of(part, aliases) for part in item["parts"])
    if kind == "capture":
        return f"${item['index']}"
    if kind == "alias_ref":
        name = item["name"]
        if name not in aliases:
            raise UnsupportedConstruct(f"unresolved alias {name}")
        raise UnsupportedConstruct("alias in a result")
    raise UnsupportedConstruct(f"item kind {kind} in a result")


def _guarded_regex(rule: dict, aliases: dict[str, str]) -> str:
    """Rule pattern wrapped with its before/after constraints as
    lookarounds (the subst-family rendering of parallel constraints)."""
    prefix = ""
    suffix = ""
    for constraint in rule["constraints"]:
        fragment = _regex_of(constraint["item"], aliases)
        if constraint["kind"] == "before":
            prefix += f"(?<={fragment})"
        elif constraint["kind"] == "not_before":
            prefix += f"(?<!{fragment})"
        elif constraint["kind"] == "after":
            suffix += f"(?={fragment})"
        elif constraint["kind"] == "not_after":
            suffix += f"(?!{fragment})"
    return prefix + _regex_of(rule["from"], aliases) + suffix


def _stage_tree(
    body_item: dict,
    dep_aliases: dict[str, str],
    aliases: dict[str, str],
) -> dict | None:
    kind = body_item["kind"]
    if kind == "parallel":
        subs = []
        capture_rules = []
        for rule in body_item["rules"]:
            if _has_capture(rule["from"]) or _has_capture(rule["to"]):
                capture_rules.append(rule)
                continue
            sub = {
                "pattern": _render_item(rule["from"], aliases),
                "result": _render_item(rule["to"], aliases),
            }
            for constraint in rule["constraints"]:
                if constraint["kind"] == "before":
                    sub["before"] = _render_item(constraint["item"], aliases)
                elif constraint["kind"] == "after":
                    sub["after"] = _render_item(constraint["item"], aliases)
            subs.append(sub)
        if capture_rules:
            # Capture-bearing rules degrade to ordered substitutions ahead
            # of the parallel pass: the parallel engine path has no group
            # vocabulary. Same observable result for the maps that use
            # this shape (digraph-prevention rules that fire first).
            raise _SplitParallel(subs, capture_rules)
        return {"kind": "parallel", "subs": subs}
    if kind == "sequence":
        # A sequence is ordered substitution: one subst per rule.
        return None  # handled by the caller (expands to multiple items)
    if kind == "run":
        dep = body_item.get("dependency")
        if dep:
            target = dep_aliases.get(dep, dep)
            return {"kind": "run", "map": target}
        raise UnsupportedConstruct("same-map stage run (run stage.X)")
    if kind in ("compose", "decompose"):
        return {"kind": kind}
    if kind == "string_case":
        return {"kind": body_item["op"]}
    raise UnsupportedConstruct(f"stage item {kind}")


_LIB_CACHE: dict[str, dict[str, str]] = {}


def _library_aliases(name: str, libs_dir: Path) -> dict[str, str]:
    """Harvest `def_alias NAME, expr` lines from an .iml library.

    The .iml expression syntax is exactly the engine's expression
    layer, so alias values are reused verbatim (expr renderer) and
    compiled through `expr_to_regex` (regex renderer) — no grammar is
    duplicated here.
    """
    if name in _LIB_CACHE:
        return _LIB_CACHE[name]
    path = libs_dir / f"{name}.iml"
    aliases: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*def_alias\s+([\w-]+)\s*,\s*(.+?)\s*$", line)
            if m:
                # Some libraries single-quote their any(...) literals; the
                # expression layer expects double quotes.
                value = m.group(2)
                if "'" in value and '"' not in value:
                    value = value.replace("'", '"')
                aliases[m.group(1)] = value
    _LIB_CACHE[name] = aliases
    return aliases


def isc_to_tree(source: str, filename: str | None = None, on_unsupported: str = "raise") -> dict:
    """Parse .isc source and convert to the engine's tree shape.

    ``on_unsupported`` mirrors the engine's contract: "raise" (default)
    fails loudly on constructs outside the engine vocabulary; "skip"
    drops the unrenderable rule and records it in the tree's
    ``skipped_unsupported``.
    """
    doc = _IscParser(source, filename).parse()
    skipped: list[str] = []

    # Library dependencies ("posix", "unicode") contribute aliases to
    # this map's scope; system dependencies do not.
    libs_dir = Path(filename).parent.parent / "libs" if filename else Path("libs")
    library_aliases: dict[str, str] = {}
    for dep in doc["dependencies"]:
        if "/" not in dep["target"] and "." not in dep["target"]:
            library_aliases.update(_library_aliases(dep["target"], libs_dir))

    aliases: dict[str, str] = dict(library_aliases)
    for alias in doc["aliases"]:
        aliases[alias["name"]] = _render_item(alias["value"], aliases)
    regex_aliases: dict[str, str] = {
        name: expr_to_regex(value) for name, value in library_aliases.items()
    }
    for alias in doc["aliases"]:
        regex_aliases[alias["name"]] = _regex_of(alias["value"], regex_aliases)

    dep_aliases = {
        d["aliasName"]: d["target"] for d in doc["dependencies"] if d.get("aliasName")
    }

    def _subst(rule: dict) -> dict:
        return {
            "kind": "subst",
            "pattern": _guarded_regex(rule, regex_aliases),
            "result": _repl_of(rule["to"], regex_aliases),
        }

    def _maybe(rule: dict, rules: list[dict]) -> None:
        """Append a rendered rule, honoring on_unsupported."""
        if on_unsupported != "skip":
            rules.append(rule())
            return
        try:
            rules.append(rule())
        except _SplitParallel as split:
            raise
        except UnsupportedConstruct as e:
            skipped.append(str(e))

    # The engine consumes a flat, ordered rule list; ISC stage names do
    # not survive conversion (same-map named runs are unsupported anyway).
    stages: list[dict] = []
    for stage in doc["stages"]:
        rules: list[dict] = []
        for body_item in stage["body"]:
            kind = body_item["kind"]
            if kind == "sequence":
                for rule in body_item["rules"]:
                    _maybe(lambda rule=rule: _subst(rule), rules)
            elif kind == "bare_rule":
                rule = body_item["rule"]
                _maybe(lambda rule=rule: _subst(rule), rules)
            else:
                try:
                    _maybe(
                        lambda body_item=body_item: _stage_item(
                            body_item, dep_aliases, aliases
                        ),
                        rules,
                    )
                except _SplitParallel as split:
                    # Capture-bearing parallel rules degrade to ordered
                    # substitutions ahead of the capture-free parallel op.
                    for rule in split.capture_rules:
                        _maybe(lambda rule=rule: _subst(rule), rules)
                    rules.append({"kind": "parallel", "subs": split.subs})
        stages.append({"children": rules})

    return {
        "metadata": doc["metadata"],
        # The engine's test consumers expect (input, expected) pairs.
        "tests": [(t["input"], t["expected"]) for t in doc["tests"]],
        "stages": stages,
        "dependencies": [d["target"] for d in doc["dependencies"]],
        "skipped_unsupported": skipped,
    }


def _stage_item(body_item: dict, dep_aliases: dict[str, str], aliases: dict[str, str]) -> dict:
    """Convert one non-rule stage item; may raise _SplitParallel."""
    return _stage_tree(body_item, dep_aliases, aliases)


def parse_isc_file(path: str | Path, on_unsupported: str = "raise") -> dict:
    """Parse a .isc file from disk into the engine's tree shape."""
    file = Path(path)
    return isc_to_tree(file.read_text(encoding="utf-8"), str(file), on_unsupported)
