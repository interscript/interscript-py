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

import json
import re
from pathlib import Path

from .expr import _BOUNDARY, _WORD, expr_lookbehind, expr_neg_lookbehind, expr_to_regex, is_plain_string

# Ruby \w is ASCII-only; the same divergence the expression layer fixes.
_WORD_BOUNDARY = _BOUNDARY
_NON_WORD_BOUNDARY = "(?:(?<=" + _WORD + ")(?=" + _WORD + ")|(?<!" + _WORD + ")(?!" + _WORD + "))"


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


PRIMITIVES = {"boundary", "line_start", "line_end", "word_boundary", "non_word_boundary", "space"}
_FUNCTIONS = {"upcase", "downcase", "title_case", "reverse", "strip", "swapcase"}
_CONSTRAINTS = {"before", "after", "not_before", "not_after"}
# Stdlib aliases usable as bare names (Ruby Stdlib::ALIASES, resolved
# before doc-local aliases, mirroring the interpreter's lookup order).
_STDLIB_EXPR = {"alpha", "digit", "word", "not_word", "any_character"}
# Tokens that terminate an item inside a rule; a bare word equal to one
# of these is a keyword, never an alias reference.
_KEYWORDS = _CONSTRAINTS | {"to", "from", "note", "when"}
# The WF-1 keyword table (grammar keyword = ... in isc.parg / the ruby
# keyword rule): guard lists and declaration lists stop at any of
# these instead of swallowing the next construct's head token.
_WF1_KEYWORDS = {
    "parallel", "sequence", "stage", "compose", "separate", "system",
    "metadata", "aliases", "tests", "notes", "description",
    "any_character", "authority", "dependency", "run", "sub",
    "before", "after", "not_before", "not_after", "any", "none",
    "boundary", "line_start", "line_end", "word_boundary", "downcase",
    "upcase", "title_case", "capture", "maybe", "some", "ref", "name",
    "ruleset", "use", "override", "languages", "when",
}

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
        # maps#293 (WO20 fuzz): the reference grammars anchor isc_source
        # at end-of-input — any content after the system block is a
        # parse error. Without this, a mis-scoped block closed the
        # system early and the rest of the document (tests, stages) was
        # silently dropped from the parsed tree.
        self._skip_ws()
        if self._pos < len(self._src):
            self._error("Unexpected content after the system block")

        doc: dict = {
            "systemCode": system_code,
            "metadata": {},
            "tests": [],
            "aliases": [],
            "stages": [],
            "dependencies": [],
            "languages": [],
            "rulesets": [],
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
            elif "languages" in item:
                doc["languages"].extend(item["languages"])
            elif "ruleset" in item:
                doc["rulesets"].append(item["ruleset"])
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
            elif keyword == "languages":
                # ISC-WF-1 W3: `languages a, b` — the languages the map
                # serves, declared once per map alongside metadata (the
                # reference grammars' languages_decl; identifiers are
                # keyword-guarded so the repeat stops at the next block
                # item).
                items.append(self._languages_decl())
            elif keyword == "ruleset":
                # ISC-WF-1 W1: a named, reusable rule set; bodies are
                # stage bodies (spec §3.1) and never execute directly —
                # they splice into stages through `use`.
                items.append(self._ruleset())
            else:
                self._error(f"Unexpected keyword: {keyword}")

    def _languages_decl(self) -> dict:
        self._expect("languages")
        self._skip_ws()
        languages: list[str] = []
        while True:
            word = self._peek_strict_ident()
            if not word or word in _WF1_KEYWORDS:
                break
            self._consume(word)
            languages.append(word)
            saved = self._pos
            self._skip_inline_ws()
            if self._peek() == ",":
                # comma_ws: a comma REQUIRES another identifier (the
                # reference grammar's *(((comma_ws / whitespace) ident))).
                self._pos += 1
                self._skip_ws()
                nxt = self._peek_strict_ident()
                if not nxt or nxt in _WF1_KEYWORDS:
                    self._error("Expected language identifier after ','")
                continue
            self._skip_ws()
            nxt = self._peek_strict_ident()
            if nxt and nxt not in _WF1_KEYWORDS:
                continue
            self._pos = saved
            break
        if not languages:
            self._error("Expected language identifier after 'languages'")
        return {"languages": languages}

    def _ruleset(self) -> dict:
        self._expect("ruleset")
        self._skip_ws()
        name = self._strict_ident()
        self._skip_ws()
        self._expect("{")
        body = self._stage_items()
        self._expect("}")
        return {"ruleset": {"name": name, "body": body}}

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
                        try:
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
                            continue
                        except IscParseError:
                            # The reference grammar's ordered choice: the
                            # note-line list is tried first; when it fails
                            # mid-block (prose notes carry unescaped inner
                            # quotes — din-urd, komaba), the generic-field
                            # production re-reads the whole block as one
                            # raw-text value ending at the first `}`.
                            self._pos = saved
                        metadata["notes"] = self._raw_text()
                        self._expect("}")
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
                # maps#293 (WO20 fuzz): grammar field_raw =
                # 1*(!newline !"}" !"{" chr) — a generic field's bare
                # value cannot contain braces, and a "{" in value
                # position stays unconsumed so the block grammar rejects
                # (e.g. a ruleset nested inside metadata — the reference
                # trio parse-reject it). py used to swallow the whole
                # line, braces included, as a string value.
                metadata[key] = self._field_raw_value()
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
        body = self._stage_items()
        self._expect("}")
        return {"stage": {"name": name, "body": body}}

    def _stage_items(self) -> list[dict]:
        """Stage/rule-set body: parallel and sequence blocks, run/separate/
        compose/decompose/string-case directives, funcalls, bare rules and
        `use` directives (WF-1)."""
        body: list[dict] = []
        while True:
            self._skip_ws()
            # Progress invariant (same guard as _block_items): EOF means
            # the block never closes — break and let the trailing
            # _expect("}") name the failure. Pre-fix the skip-line
            # fallback below span forever on truncated maps
            # (interscript-py#43, the WO20 fuzz campaign's hang class).
            if self._peek() == "}" or self._pos >= len(self._src):
                break
            if self._peek() == "#":
                self._skip_line()
                continue
            keyword = self._peek_word()
            if keyword in ("parallel", "sequence"):
                self._consume(keyword)
                self._skip_ws()
                self._expect("{")
                rules = self._rule_lines()
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
                item = {"kind": "string_case", "op": keyword}
                self._skip_inline_ws()
                if keyword == "title_case" and self._peek_word() == "word_separator":
                    self._consume("word_separator")
                    self._skip_ws()
                    self._expect(":")
                    self._skip_ws()
                    item["word_separator"] = self._item_atom()["value"]
                body.append(item)
            elif keyword == "sub":
                body.append({"kind": "bare_rule", "rule": self._rule()})
            elif keyword == "use":
                # ISC-WF-1 W1/W2: `use <ident>` (same-map ruleset) or
                # `use <alias>.<ident>` (dependency-declared), with an
                # optional override block; expanded at conversion time.
                body.append(self._use_directive())
            else:
                # maps#293 (WO20 fuzz): an unrecognized keyword in
                # stage-item position fails the grammar's stage_item
                # production. py used to silently drop the line, so a
                # typo'd or foreign directive parsed to a DIFFERENT rule
                # set than the reference trio's parse rejection.
                self._error(f"Unexpected {keyword!r} in stage item position")
        return body

    def _rule_lines(self) -> list[dict]:
        """One parallel/sequence/override block's rule lines: rule defs,
        `use` directives (§3.2 — the referenced content joins the
        enclosing pass) and comments; anything else is the strict-parse
        rejection (maps#293)."""
        rules: list[dict] = []
        while True:
            self._skip_ws()
            # Same EOF progress invariant as the stage loop.
            if self._peek() == "}" or self._pos >= len(self._src):
                break
            if self._peek() == "#":
                self._skip_line()
                continue
            if self._peek_word() == "sub":
                rules.append(self._rule())
            elif self._peek_word() == "use":
                rules.append(self._use_directive())
            else:
                keyword = self._peek_word()
                self._error(f"Unexpected {keyword!r} in rule position")
        return rules

    def _use_directive(self) -> dict:
        self._expect("use")
        self._skip_ws()
        alias = self._strict_ident()
        qualified = None
        if self._peek() == ".":
            self._pos += 1
            qualified = self._strict_ident()
        overrides = None
        self._skip_ws()
        if self._peek_word() == "override":
            self._consume("override")
            self._skip_ws()
            self._expect("{")
            overrides = self._rule_lines()
            self._expect("}")
        item: dict = {"kind": "use", "alias": alias}
        if qualified is not None:
            item["qualified"] = qualified
        if overrides is not None:
            item["overrides"] = overrides
        return item

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
                elif word == "when":
                    # ISC-WF-1 W3: a language guard on the rule (the
                    # guard fires only when that language is selected).
                    rule["when"] = self._when_guard()
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
        if self._peek_word() == "when":
            rule = {"from": from_item, "to": to_item, "constraints": constraints}
            rule["when"] = self._when_guard()
            return rule
        return {"from": from_item, "to": to_item, "constraints": constraints}

    def _when_guard(self) -> list[str]:
        """`when <ident> (","|ws <ident>)*` — language names are
        keyword-guarded so the repeat stops at the next rule/block head
        instead of swallowing it."""
        self._expect("when")
        self._skip_ws()
        languages = [self._strict_ident()]
        while True:
            saved = self._pos
            self._skip_inline_ws()
            if self._peek() == ",":
                self._pos += 1
                self._skip_ws()
                nxt = self._peek_strict_ident()
                if not nxt or nxt in _WF1_KEYWORDS:
                    self._error("Expected language identifier after ','")
                languages.append(nxt)
                continue
            self._skip_ws()
            nxt = self._peek_strict_ident()
            if nxt and nxt not in _WF1_KEYWORDS:
                languages.append(nxt)
                continue
            self._pos = saved
            break
        return languages

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
        if self._src.startswith(".", self._pos):
            self._pos += 1
            self._skip_inline_ws()
            name = self._identifier()
            return {"type": "alias_ref", "name": name, "map": word}
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
                    result += self._unicode_escape()
                elif nxt in {"n", "t", "r", '"', "\\"}:
                    result += {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}[nxt]
                elif nxt == "U":
                    result += self._capital_unicode_escape()
                else:
                    # maps#293 (WO20 fuzz): the grammar's escape_sequence
                    # admits only n r t double-quote backslash uXXXX and
                    # the \U forms — anything else fails the string
                    # production (ruby: "expected one of ['n','r','t','\"',
                    # '\\','u'...]"). py used to pass the letter through,
                    # accepting malformed rule strings the trio reject.
                    self._error(
                        f"Invalid escape \\{nxt}: expected one of n, r, t, quote, backslash, u, U",
                        self._pos - 2,
                    )
            else:
                result += c
        self._error("Unterminated string")

    def _capital_unicode_escape(self) -> str:
        """Decode ``\\U`` per the grammar's ordered escape alternatives
        (``grammar/isc.parg``): esc_uni = ``"U" (4*8hexd) as unicode``
        decodes 4 to 8 hex digits, greedily, to one code point — the
        WO19 G6 ruling (BOTH spellings decode; ruby/rs always did, ts
        since the gki-bel ``"\\U040E"`` fix). With fewer than 4 hex
        digits esc_uni fails and esc_u_lone renders a literal ``"U"``;
        the digits then stay raw text (measured on ruby 4/4:
        ``"\\Uab"`` → ``Uab``). The first 8 hex digits are consumed, so
        a 9th remains raw (``"\\U000000410"`` → ``A0``, ruby-verified).

        The capital form was previously rendered as a bare ``"U"`` with
        the digits as raw text — a stale 0.1.x ts hand-parser parity
        note (ts decodes since the G6 fix), which left the three
        Belarusian uppercase-Ў rows (gki-bel-2000 / mvd-bel-2008 /
        odni-bel-2015: ``sub "\\U040E" "W"-style`` rules) unfired and
        diverging from the golden corpus.
        """
        start = self._pos
        digits = self._src[start : start + 8]
        n = 0
        while n < len(digits) and digits[n] in "0123456789abcdefABCDEF":
            n += 1
        if n < 4:
            # esc_u_lone: a lone "U"; the raw digits (if any) are
            # consumed as ordinary text by the main loop.
            return "U"
        self._pos = start + n
        return chr(int(digits[:n], 16))

    def _unicode_escape(self) -> str:
        """Decode ``\\uXXXX`` — the grammar's esc_uni4 (exactly 4 hex
        digits, ``grammar/isc.parg``). Any other form (short hex,
        non-hex digits, the braced form, end of input) is a parse error
        at the escape's position: the pipeline used to leak a raw
        ``ValueError`` from ``int()`` here (interscript-py#44, the WO20
        fuzz campaign's largest crash class — it fires on note text too,
        which shares this scanner).

        Unlike ``\\u`` (esc_uni4, exactly 4 — no later alternative can
        match a lowercase ``u``, so a short form fails the string
        production), the capital ``\\U`` falls back to esc_u_lone with
        fewer than 4 hex digits — see ``_capital_unicode_escape``.
        """
        slash = self._pos - 2  # the backslash, for the error position
        digits = self._src[self._pos : self._pos + 4]
        if len(digits) < 4 or any(c not in "0123456789abcdefABCDEF" for c in digits):
            self._error("Invalid \\u escape: expected exactly 4 hexadecimal digits", slash)
        self._pos += 4
        return chr(int(digits, 16))

    def _single_quoted(self) -> str:
        self._expect("'")
        start = self._pos
        while self._pos < len(self._src) and self._src[self._pos] != "'":
            self._pos += 1
        result = self._src[start : self._pos]
        self._expect("'")
        return result

    def _raw_text(self) -> str:
        # maps#295 (WO20 fuzz): free text runs to the FIRST unescaped "}"
        # — grammar raw_text = *( %x5C "{" / %x5C "}" / (!"}" chr) ), the
        # exact scan the reference trio perform. A bare "{" in prose is
        # ORDINARY TEXT: brace-depth tracking made a single "{" inside a
        # description swallow the rest of the document (notes, tests, the
        # stage block) into a metadata string, so a one-character prose
        # edit silently changed the parsed rule set and the output.
        result = ""
        while self._pos < len(self._src):
            c = self._src[self._pos]
            if c == "\\" and self._pos + 1 < len(self._src) and self._src[self._pos + 1] in "{}":
                result += self._src[self._pos + 1]
                self._pos += 2
                continue
            if c == "}":
                break
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

    def _strict_ident(self) -> str:
        """The reference grammar's identifier: [A-Za-z_][A-Za-z0-9_]* —
        used by the WF-1 productions (ruleset/use/when/languages names),
        which are ASCII-strict unlike the legacy reader above."""
        self._skip_ws()
        ident = self._peek_strict_ident()
        if not ident:
            self._error("Expected identifier")
        self._pos += len(ident)
        return ident

    def _peek_strict_ident(self) -> str:
        start = self._pos
        if start >= len(self._src):
            return ""
        c = self._src[start]
        if not (c.isascii() and (c.isalpha() or c == "_")):
            return ""
        i = start + 1
        while i < len(self._src):
            c = self._src[i]
            if c.isascii() and (c.isalnum() or c == "_"):
                i += 1
            else:
                break
        return self._src[start:i]

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

    def _field_raw_value(self) -> str:
        """Grammar field_raw = 1*(!newline !"}" !"{" chr): a generic
        metadata field's bare value ends at the newline and cannot
        contain braces. Braces stay unconsumed so the block grammar
        names the failure (maps#293)."""
        start = self._pos
        while self._pos < len(self._src) and self._src[self._pos] not in "{}\n":
            self._pos += 1
        if self._pos == start:
            self._error("Expected field value")
        return self._src[start : self._pos].strip()

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

    def _error(self, message: str, pos: int | None = None) -> None:
        at = self._pos if pos is None else pos
        line = self._src.count("\n", 0, at) + 1
        col = at - (self._src.rfind("\n", 0, at) + 1) + 1
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
        if all(sub["type"] == "string" for sub in items) and all(
            len(sub["value"]) == 1 for sub in items
        ):
            return f'any("{_escape("".join(sub["value"] for sub in items))}")'
        # any(<single alias resolving to a plain string>) is a character
        # class (measured against Ruby: the unicode library's greek);
        # a bare alias reference stays a literal sequence. Imported
        # aliases resolve through the qualified path.
        if (
            len(items) == 1
            and items[0]["type"] == "alias_ref"
            and not items[0].get("map")
        ):
            name = items[0]["name"]
            if name in aliases:
                resolved = aliases[name]
            elif name in _imported:
                resolved = _qualified_expr(None, name)
            else:
                resolved = None
            if resolved is not None and is_plain_string(resolved):
                return f"any({resolved})"
        return "any([" + ", ".join(_render_item(sub, aliases) for sub in items) + "])"
    if kind == "range":
        return f'any("{_escape(item["lo"])}".."{_escape(item["hi"])}")'
    if kind == "capture_group":
        return "capture(" + _render_item(item["inner"], aliases) + ")"
    if kind == "maybe":
        return "maybe(" + _render_item(item["inner"], aliases) + ")"
    if kind == "some":
        return "some(" + _render_item(item["inner"], aliases) + ")"
    if kind == "primitive":
        name = item["name"]
        if name == "space":
            return "space"
        if name == "boundary":
            return "boundary"
        if name == "non_word_boundary":
            return "non_word_boundary"
        if name == "line_start":
            return "line_start"
        if name == "line_end":
            return "line_end"
        raise UnsupportedConstruct(f"primitive {name}")
    if kind == "function" and item["name"] == "any_character":
        return "any_character"
    if kind == "alias_ref":
        name = item["name"]
        if item.get("map"):
            return _qualified_expr(item["map"], name)
        if name in _STDLIB_EXPR:
            return name
        if name in _imported:
            return _qualified_expr(None, name)
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
        if all(sub["type"] == "string" for sub in items) and all(
            len(sub["value"]) == 1 for sub in items
        ):
            return "[" + re.escape("".join(sub["value"] for sub in items)) + "]"
        return "(?:" + "|".join(_regex_of(sub, aliases) for sub in items) + ")"
    if kind == "range":
        return "[" + re.escape(item["lo"]) + "-" + re.escape(item["hi"]) + "]"
    if kind == "maybe":
        return "(?:" + _regex_of(item["inner"], aliases) + ")?"
    if kind == "some":
        return "(?:" + _regex_of(item["inner"], aliases) + ")+"
    if kind == "capture_group":
        return "(" + _regex_of(item["inner"], aliases) + ")"
    if kind == "capture":
        # ref(N) in a from position is a backreference.
        return f"\\{item['index']}"
    if kind == "primitive":
        return {
            "boundary": _WORD_BOUNDARY,
            "non_word_boundary": _NON_WORD_BOUNDARY,
            "line_start": "^",
            "line_end": "$",
            "space": " ",
        }.get(item["name"], None) or _unsupported_primitive(item["name"])
    if kind == "function" and item["name"] == "any_character":
        return "."
    if kind == "alias_ref":
        name = item["name"]
        if item.get("map"):
            return _qualified_regex(item["map"], name)
        if name in _STDLIB_EXPR:
            return {"alpha": "[a-zA-Z]", "digit": "[0-9]", "word": "[a-zA-Z0-9_]", "not_word": "[^a-zA-Z0-9_]", "any_character": "."}[name]
        if name in _imported:
            return _qualified_regex(None, name)
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
    if kind == "capture_group":
        # Ruby renders a capture group in a result position as the
        # parenthesised inner text (bgnpcgn-zho: to capture("me, yao")).
        return "(" + _repl_of(item["inner"], aliases) + ")"
    if kind == "set":
        # A to-position any() lists alternative spellings; the Ruby
        # runtime picks the first in non-iterating mode.
        if not item["items"]:
            return ""
        return _repl_of(item["items"][0], aliases)
    if kind == "alias_ref":
        name = item["name"]
        if item.get("map"):
            return _qualified_repl(item["map"], name)
        if name in _imported:
            return _qualified_repl(None, name)
        if name not in aliases:
            raise UnsupportedConstruct(f"unresolved alias {name}")
        raise UnsupportedConstruct("alias in a result")
    if kind == "primitive":
        if item["name"] == "space":
            return " "
        raise UnsupportedConstruct(f"primitive {item['name']} in a result")
    raise UnsupportedConstruct(f"item kind {kind} in a result")


def _guarded_regex(rule: dict, aliases: dict[str, str], expr_aliases: dict[str, str]) -> str:
    """Rule pattern wrapped with its before/after constraints as
    lookarounds (the subst-family rendering of parallel constraints).
    Guards render through the expression layer so mixed-width
    lookbehinds distribute over branches (Python re is stricter than
    Onigmo here)."""
    prefix = ""
    suffix = ""
    for constraint in rule["constraints"]:
        if constraint["kind"] == "before":
            prefix += expr_lookbehind(_render_item(constraint["item"], expr_aliases))
        elif constraint["kind"] == "not_before":
            prefix += expr_neg_lookbehind(_render_item(constraint["item"], expr_aliases))
        elif constraint["kind"] == "after":
            suffix += "(?=" + _regex_of(constraint["item"], aliases) + ")"
        elif constraint["kind"] == "not_after":
            suffix += "(?!" + _regex_of(constraint["item"], aliases) + ")"
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
            if rule.get("when"):
                # ISC-WF-1 W3: the guard rides on the sub; the engine
                # resolves it per selected language (§5.1/§5.2).
                sub["when"] = rule["when"]
            for constraint in rule["constraints"]:
                if constraint["kind"] == "before":
                    sub["before"] = _render_item(constraint["item"], aliases)
                elif constraint["kind"] == "not_before":
                    sub["not_before"] = _render_item(constraint["item"], aliases)
                elif constraint["kind"] == "after":
                    sub["after"] = _render_item(constraint["item"], aliases)
                elif constraint["kind"] == "not_after":
                    sub["not_after"] = _render_item(constraint["item"], aliases)
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
            op = {"kind": "run", "map": target}
            if body_item.get("stage"):
                op["stage"] = body_item["stage"]
            return op
        return {"kind": "run_stage", "name": body_item["stage"]}
    if kind in ("compose", "decompose"):
        return {"kind": kind}
    if kind == "string_case":
        op = {"kind": body_item["op"]}
        if "word_separator" in body_item:
            op["word_separator"] = body_item["word_separator"]
        return op
    if kind == "separate":
        sep = {"kind": "separate"}
        separator = body_item.get("separator")
        if separator and separator.get("type") == "string":
            sep["separator"] = separator["value"]
        return sep
    if kind == "funcall":
        # rababa and friends: the trio load these fine and name the
        # external dependency at EXECUTION time (ruby: ExternalUtilError
        # without the Rababa gem). Same contract here — a named
        # unsupported op at exec, never a load-time rejection.
        return {
            "kind": "unsupported",
            "what": (
                f"funcall {body_item.get('name')} (external function call; "
                "the callee is not part of this runtime's vocabulary)"
            ),
        }
    raise UnsupportedConstruct(f"stage item {kind}")


_LIB_CACHE: dict[str, dict[str, str]] = {}

# Dep-alias context of the document being converted, and a memoized
# {target: {alias-name: item}} table for qualified alias resolution
# (map.<dep-alias>.<name>). Single-threaded conversion; reset per doc.
_dep_aliases: dict[str, str] = {}
_DEP_ALIAS_ITEMS: dict[str, dict] = {}
_imported: dict[str, str] = {}
# Memoized parsed dependency documents, keyed by resolved path (WF-1
# ruleset splices and alias resolution share the parse).
_DEP_DOCS: dict[str, dict] = {}


def _is_library_target(target: str) -> bool:
    # Libraries carry no path separators or dots (same heuristic as the
    # library-alias harvester) and resolve through _library_aliases.
    return "/" not in target and "." not in target


def _dep_doc_cached(key: str, path) -> dict:
    if key not in _DEP_DOCS:
        _DEP_DOCS[key] = _IscParser(Path(path).read_text(encoding="utf-8"), str(path)).parse()
    return _DEP_DOCS[key]


def _dep_items(target: str) -> dict:
    """Parse a dependency map's source and return its alias items,
    memoized per target."""
    if target in _DEP_ALIAS_ITEMS:
        return _DEP_ALIAS_ITEMS[target]
    from .interscript import _find_map

    path = _find_map(target)
    if path is None:
        raise UnsupportedConstruct(f"dependency map {target!r} not found")
    doc = _dep_doc_cached(str(path), path)
    items = {a["name"]: a["value"] for a in doc["aliases"]}
    _DEP_ALIAS_ITEMS[target] = items
    return items


def _qualified_items(map_alias: str, name: str) -> tuple[dict, dict]:
    target = _dep_aliases.get(map_alias) or _imported.get(name) or _dep_aliases.get(name)
    if target is None:
        raise UnsupportedConstruct(f"unresolved alias {name}")
    return _qualified_items_for(target, name)


def _qualified_items_for(target: str, name: str) -> tuple[dict, dict]:
    _dep_aliases = target
    items = _dep_items(target)
    if name not in items:
        raise UnsupportedConstruct(f"unresolved alias {target}.{name}")
    return items[name], items


def _qualified_expr(map_alias: str, name: str) -> str:
    item, items = _qualified_items(map_alias, name)
    return _render_item(item, {k: _render_item(v, items) for k, v in items.items()})


def _qualified_regex(map_alias: str, name: str) -> str:
    item, _ = _qualified_items(map_alias, name)
    return expr_to_regex(_qualified_expr(map_alias, name))


def _qualified_repl(map_alias: str, name: str) -> str:
    item, _ = _qualified_items(map_alias, name)
    return _repl_of(item, {})


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
                # Library aliases may reference aliases defined earlier
                # in the same file (var-kor: jamo = any([jamo_leading_cons,
                # ...])); substitute them, outside string literals only.
                value = _subst_alias_refs(value, aliases)
                # A library alias holding a bare string acts as a
                # character class inside any() (measured: the unicode
                # library's greek); an in-map string alias would be a
                # literal sequence instead.
                if re.fullmatch(r'"(?:[^"\\]|\\.)*"', value):
                    value = f"any({value})"
                aliases[m.group(1)] = value
    _LIB_CACHE[name] = aliases
    return aliases


def _subst_alias_refs(value: str, resolved: dict[str, str]) -> str:
    out, i, n = [], 0, len(value)
    while i < n:
        c = value[i]
        if c == '"':
            j = i + 1
            while j < n:
                if value[j] == "\\":
                    j += 2
                    continue
                if value[j] == '"':
                    j += 1
                    break
                j += 1
            out.append(value[i:j])
            i = j
        elif c.isascii() and (c.isalnum() or c == "_"):
            j = i
            while j < n and (value[j].isascii() and (value[j].isalnum() or value[j] in "_-")):
                j += 1
            word = value[i:j]
            out.append(resolved.get(word, word))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _rule_from_key(rule: dict) -> str:
    """Structural key of a rule's from-item — the byte-exact from
    identity of the W2 override merge (§4.2; ruby from.inspect / ts
    itemKey analog). Two identically written rules produce identical
    keys; no normalization."""
    return json.dumps(rule["from"], sort_keys=True, ensure_ascii=False)


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
    lib_names: set[str] = set()
    for dep in doc["dependencies"]:
        if "/" not in dep["target"] and "." not in dep["target"]:
            if (libs_dir / f"{dep['target']}.iml").is_file():
                lib_names.add(dep["target"])
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

    _dep_aliases.clear()
    _DEP_ALIAS_ITEMS.clear()
    _imported.clear()
    _dep_aliases.update(dep_aliases)

    # Alias-less (import) dependencies merge their aliases into scope.
    # A target that resolves to a map file is parsed; legacy libraries
    # were already harvested into library_aliases above.
    for d in doc["dependencies"]:
        if d.get("aliasName"):
            continue
        if d["target"] in lib_names:
            # A library dependency (posix, unicode) contributes through
            # the harvest above; an emitted map of the same name in a
            # load path must not hijack it into the map-import path.
            continue
        from .interscript import _find_map

        if _find_map(d["target"]) is not None:
            items = _dep_items(d["target"])
            for name in items:
                _imported.setdefault(name, d["target"])

    # -- ISC-WF-1 W1/W2: ruleset expansion --------------------------------
    # `use` is a compile-time macro: the referenced ruleset's expanded
    # items splice in at the use site (ruby node_adapter expand_use / ts
    # expand.ts). Owner context = (systemCode, rulesets, dep-aliases) of
    # the document owning the items being expanded — nested `use` items
    # resolve against their OWNER, not this document.

    doc_rulesets = {r["name"]: r["body"] for r in doc["rulesets"]}

    def _owner_rulesets(target: str) -> tuple[str, dict, dict]:
        """The (systemCode, rulesets, dep-aliases) of a dependency doc,
        resolved through the load paths or the corpus's sibling libs/ dir
        (the trio's load_path carries both; py CI pins maps/ only)."""
        from .interscript import _find_map

        path = _find_map(target)
        if path is None:
            cand = libs_dir / f"{target}.isc"
            if cand.is_file():
                path = cand
        if path is None:
            raise UnsupportedConstruct(
                f"use: cannot resolve dependency {target!r} (needed for its rulesets)"
            )
        dep_doc = _dep_doc_cached(str(path), path)
        dep_rulesets = {r["name"]: r["body"] for r in dep_doc["rulesets"]}
        dep_dep_aliases = {
            d["aliasName"]: d["target"] for d in dep_doc["dependencies"] if d.get("aliasName")
        }
        return dep_doc["systemCode"], dep_rulesets, dep_dep_aliases

    def expand_items(items: list, chain: list, owner: tuple) -> list:
        out = []
        for it in items:
            kind = it.get("kind") if isinstance(it, dict) else None
            if kind == "use":
                out.extend(expand_use(it, chain, owner))
            elif kind in ("parallel", "sequence"):
                # §3.2: a `use` row splices the referenced ruleset's rule
                # content into the enclosing pass (groups flatten).
                rules = []
                for r in it["rules"]:
                    if r.get("kind") == "use":
                        rules.extend(expand_use_flat(r, chain, owner))
                    else:
                        rules.append(r)
                out.append({**it, "rules": rules})
            else:
                out.append(it)
        return out

    def expand_use(use_item: dict, chain: list, owner: tuple) -> list:
        syscode, rulesets, owner_deps = owner
        alias = use_item["alias"]
        qualified = use_item.get("qualified")
        if qualified:
            target = owner_deps.get(alias)
            if target is None:
                raise UnsupportedConstruct(
                    f"use {alias}.{qualified}: no declared dependency aliased {alias!r}"
                )
            dep_syscode, dep_rulesets, dep_deps = _owner_rulesets(target)
            body = dep_rulesets.get(qualified)
            if body is None:
                raise UnsupportedConstruct(
                    f"use {alias}.{qualified}: ruleset not found in dependency {target!r}"
                )
            key = f"{dep_syscode}::{qualified}"
            inner_owner = (dep_syscode, dep_rulesets, dep_deps)
        else:
            body = rulesets.get(alias)
            if body is None:
                raise UnsupportedConstruct(f"use {alias}: ruleset not found")
            key = f"{syscode}::{alias}"
            inner_owner = owner
        if key in chain:
            raise UnsupportedConstruct(f"use cycle: {' -> '.join(chain + [key])}")
        expanded = expand_items(body, chain + [key], inner_owner)
        overrides = use_item.get("overrides")
        if overrides:
            # W2 (§4.2): child rules win on exact from-equality; the
            # override rows join the expansion's trailing parallel block
            # when there is one (§4.2.5), else trail as bare rules.
            o_rows: list[dict] = []
            for r in overrides:
                if r.get("kind") == "use":
                    o_rows.extend(expand_use_flat(r, chain, inner_owner))
                else:
                    o_rows.append(r)
            o_keys = {_rule_from_key(r) for r in o_rows}
            kept = []
            for it in expanded:
                it_kind = it.get("kind")
                if it_kind == "bare_rule":
                    if _rule_from_key(it["rule"]) not in o_keys:
                        kept.append(it)
                elif it_kind in ("parallel", "sequence"):
                    kept.append(
                        {**it, "rules": [r for r in it["rules"] if _rule_from_key(r) not in o_keys]}
                    )
                else:
                    kept.append(it)
            tail = kept[-1] if kept else None
            if tail is not None and tail.get("kind") == "parallel":
                kept[-1] = {**tail, "rules": tail["rules"] + o_rows}
            else:
                kept.extend({"kind": "bare_rule", "rule": r} for r in o_rows)
            expanded = kept
        return expanded

    def expand_use_flat(use_item: dict, chain: list, owner: tuple) -> list:
        rules = flatten_rules(expand_use(use_item, chain, owner))
        overrides = use_item.get("overrides")
        if not overrides:
            return rules
        o_rows = []
        for r in overrides:
            if r.get("kind") == "use":
                o_rows.extend(expand_use_flat(r, chain, owner))
            else:
                o_rows.append(r)
        o_keys = {_rule_from_key(r) for r in o_rows}
        # §4.3 flat merge: matching rules drop wherever they sit, the
        # override rows land at the tail; the caller re-wraps the list
        # into the enclosing block's structure.
        return [r for r in rules if _rule_from_key(r) not in o_keys] + o_rows

    def flatten_rules(items: list) -> list:
        rules = []
        for it in items:
            kind = it.get("kind")
            if kind in ("parallel", "sequence"):
                rules.extend(it["rules"])
            elif kind == "bare_rule":
                rules.append(it["rule"])
            else:
                raise UnsupportedConstruct(
                    "use inside a parallel/sequence block: the referenced ruleset "
                    "carries non-rule items that cannot join the pass"
                )
        return rules

    main_owner = (doc["systemCode"], doc_rulesets, dep_aliases)
    expanded_bodies: dict[str, list] = {}

    def _subst(rule: dict) -> dict:
        try:
            pattern = _guarded_regex(rule, regex_aliases, aliases)
        except UnsupportedConstruct as e:
            # maps#293 (WO20 fuzz): a zero-width primitive inside a
            # compound from (word_boundary + capture(some("z")) +
            # word_boundary) is a GRAMMAR-legal document — the reference
            # trio parse it and name the primitive at execution ("Alias
            # word_boundary not found" / "unknown primitive", the logic
            # bucket). The failure moves to the engine's unsupported-op
            # path: a named ExecutionError at exec, recorded-and-skipped
            # under on_unsupported="skip" — never a conversion-time
            # rejection of a document the trio accept.
            if str(e).startswith("primitive "):
                return {"kind": "unsupported", "what": str(e)}
            raise
        to = rule["to"]
        if to.get("type") == "function" and to.get("name") in _FUNCTIONS:
            # `to upcase` and friends: the replacement is the match
            # itself, passed through the casing function.
            op = {
                "kind": "subst",
                "pattern": pattern,
                "result": "",
                "case": to["name"],
            }
            if rule.get("when"):
                op["when"] = rule["when"]
            return op
        op = {
            "kind": "subst",
            "pattern": pattern,
            "result": _repl_of(to, regex_aliases),
        }
        if rule.get("when"):
            op["when"] = rule["when"]
        return op

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

    # The engine consumes a flat, ordered rule list. Same-map named
    # runs (run stage.X) are inlined at conversion time, so each
    # converted stage is self-contained.
    stages_by_name = {stage["name"]: stage for stage in doc["stages"]}
    converted: dict[str, list[dict]] = {}

    def convert_stage(name: str, visiting: set[str]) -> list[dict]:
        if name in converted:
            return converted[name]
        if name in visiting:
            raise UnsupportedConstruct(f"cyclic run stage.{name}")
        visiting.add(name)
        if name not in expanded_bodies:
            expanded_bodies[name] = expand_items(
                stages_by_name[name]["body"], [], main_owner
            )
        rules: list[dict] = []
        for body_item in expanded_bodies[name]:
            kind = body_item["kind"]
            if kind == "sequence":
                for rule in body_item["rules"]:
                    _maybe(lambda rule=rule: _subst(rule), rules)
            elif kind == "bare_rule":
                rule = body_item["rule"]
                _maybe(lambda rule=rule: _subst(rule), rules)
            else:
                try:
                    op = _stage_item(body_item, dep_aliases, aliases)
                except _SplitParallel as split:
                    # Capture-bearing parallel rules degrade to ordered
                    # substitutions ahead of the capture-free parallel op.
                    for rule in split.capture_rules:
                        _maybe(lambda rule=rule: _subst(rule), rules)
                    rules.append({"kind": "parallel", "subs": split.subs})
                    continue
                except UnsupportedConstruct as e:
                    # maps#293 (WO20 fuzz): same zero-width deferral as
                    # _subst, for the parallel-block render path (a
                    # primitive in any rule's from fails the whole pass
                    # at execution on the trio too).
                    if str(e).startswith("primitive "):
                        op = {"kind": "unsupported", "what": str(e)}
                    else:
                        raise
                if op is None:
                    continue
                if op.get("kind") == "run_stage":
                    if op["name"] in stages_by_name:
                        rules.extend(convert_stage(op["name"], visiting))
                    else:
                        # A stage from an imported map (alias-less
                        # dependency): run the whole imported map —
                        # import semantics merge its stages into scope.
                        target = next(
                            (
                                d["target"]
                                for d in doc["dependencies"]
                                if not d.get("aliasName")
                            ),
                            None,
                        )
                        if target is None:
                            raise UnsupportedConstruct(
                                f"unresolved stage run stage.{op['name']}"
                            )
                        rules.append({"kind": "run", "map": target})
                else:
                    rules.append(op)
        visiting.discard(name)
        converted[name] = rules
        return rules

    stages = [
        {"name": name, "children": convert_stage(name, set())}
        for name in stages_by_name
    ]

    tree = {
        "metadata": doc["metadata"],
        # The engine's test consumers expect (input, expected) pairs.
        "tests": [(t["input"], t["expected"]) for t in doc["tests"]],
        "stages": stages,
        "dependencies": [d["target"] for d in doc["dependencies"]],
        "skipped_unsupported": skipped,
    }
    if doc["languages"]:
        # ISC-WF-1 W3: the map's declared languages; the engine consults
        # them when a transliteration selects one (ruby: LanguagesDecl /
        # select_language — an undeclared selection is inert).
        tree["languages"] = doc["languages"]
    return tree


def _stage_item(body_item: dict, dep_aliases: dict[str, str], aliases: dict[str, str]) -> dict:
    """Convert one non-rule stage item; may raise _SplitParallel."""
    return _stage_tree(body_item, dep_aliases, aliases)


def parse_isc_file(path: str | Path, on_unsupported: str = "raise") -> dict:
    """Parse a .isc file from disk into the engine's tree shape."""
    file = Path(path)
    return isc_to_tree(file.read_text(encoding="utf-8"), str(file), on_unsupported)
