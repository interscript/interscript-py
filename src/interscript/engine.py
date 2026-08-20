"""Executor for parsed Interscript maps.

Semantics follow interscript-ruby for the covered op set:
- parallel { sub a, b }: ONE pass over the text; at each position the
  longest matching pattern wins. Uppercase source characters map to
  uppercased results (я->ya implies Я->YA).
- subst /pat/, res: regex substitution ($1 backreferences converted).
- run "map": apply another map's stages to the whole text.
- compose/decompose: NFC/NFD. downcase/upcase/titlecase: Unicode casing.

Unsupported constructs raise ExecutionError (on_unsupported="raise",
default) or are skipped and recorded (on_unsupported="skip").
"""

from __future__ import annotations

import re
import unicodedata

from .expr import expr_to_literal, expr_to_regex, is_plain_string


class ExecutionError(ValueError):
    """The map uses a construct this engine does not implement yet."""


def _compile_parallel(subs: list[dict]) -> tuple[re.Pattern[str], dict[str, str], dict[str, str]]:
    """Compile one parallel group: longest-pattern-first alternation
    with a named group per sub; lookaround guards for before:/after:.
    Plain-string patterns additionally feed the casing maps."""
    indexed = []
    anchor_results: dict[str, str] = {}
    n = len(subs)
    for i, sub in enumerate(subs):
        pat = expr_to_regex(sub["pattern"])
        full = pat
        if sub.get("before"):
            full = "(?<=" + expr_to_regex(sub["before"]) + ")" + full
        if sub.get("after"):
            full = full + "(?=" + expr_to_regex(sub["after"]) + ")"
        indexed.append((pat, full, f"s{i}"))
        if is_plain_string(sub["pattern"]) and not sub.get("before") and not sub.get("after"):
            src = expr_to_literal(sub["pattern"])
            if src.upper() != src:
                anchor_results[f"a{i}"] = expr_to_literal(sub["result"])
                indexed.append((re.escape(src.upper()), re.escape(src.upper()), f"a{i}"))
    indexed.sort(key=lambda t: -len(t[0]))
    combined = "|".join(f"(?P<{name}>{full})" for _, full, name in indexed)
    pattern = re.compile(combined) if indexed else re.compile(r"(?!)")

    results = {f"s{i}": expr_to_literal(sub["result"]) for i, sub in enumerate(subs)}
    results.update(anchor_results)
    casing_map: dict[str, str] = {}
    upper_dst: dict[str, str] = {}
    for sub in subs:
        if is_plain_string(sub["pattern"]) and not sub.get("before") and not sub.get("after"):
            src = expr_to_literal(sub["pattern"])
            dst = expr_to_literal(sub["result"])
            casing_map[src] = dst
            if src.upper() != src:
                upper_dst[src.upper()] = dst
    return pattern, {"casing": casing_map, "upper": upper_dst, "results": results}, {}


class Engine:
    def __init__(self, tree: dict, loader=None, on_unsupported: str = "raise") -> None:
        self.tree = tree
        self.metadata = tree.get("metadata", {})
        self._loader = loader
        self.on_unsupported = on_unsupported
        self.skipped_unsupported: list[str] = []
        self._compiled: re.Pattern[str] | None = None
        self._compiled_map: dict[str, str] = {}
        self._group_results: dict[str, str] = {}
        self._compiled_source: int | None = None

    def transliterate(self, text: str) -> str:
        for stage in self.tree.get("stages", []):
            text = self._run_stage(stage, text)
        return text

    def _run_stage(self, stage: dict, text: str) -> str:
        for child in stage.get("children", []):
            text = self._run_op(child, text)
        return text

    def _group_repl(self, m: re.Match[str], text: str) -> str:
        name = m.lastgroup if m.lastgroup else ""
        if name in self._group_results:
            result = self._group_results[name]
            tok = m.group(0)
            if result != result.upper() and tok == tok.upper() and tok != tok.lower():
                ws, we = m.start(), m.end()
                while ws > 0 and text[ws - 1].isalpha():
                    ws -= 1
                while we < len(text) and text[we].isalpha():
                    we += 1
                if text[ws:we].isupper():
                    return result.upper()
            return result
        return self._parallel_repl(m, text)

    def _parallel_repl(self, m: re.Match[str], text: str) -> str:
        tok = m.group(0)
        dst = self._compiled_map.get(tok) or self._upper_dst.get(tok)
        if dst is None:
            return tok
        # interscript-ruby casing convention: inside an ALL-CAPS source
        # word, a fully-uppercase source token uppercases its result
        # (Я -> Ya normally, YA inside БЯГА).
        if dst != dst.upper() and tok == tok.upper() and tok != tok.lower():
            ws, we = m.start(), m.end()
            while ws > 0 and text[ws - 1].isalpha():
                ws -= 1
            while we < len(text) and text[we].isalpha():
                we += 1
            if text[ws:we].isupper():
                return dst.upper()
        return dst

    def _run_op(self, op: dict, text: str) -> str:
        kind = op.get("kind")
        if kind == "parallel":
            if self._compiled is None or self._compiled_source != id(op):
                pattern, maps, _ = _compile_parallel(op["subs"])
                self._compiled = pattern
                self._compiled_map = maps["casing"]
                self._upper_dst = maps["upper"]
                self._group_results = maps["results"]
                self._compiled_source = id(op)
            return self._compiled.sub(lambda m: self._group_repl(m, text), text)
        if kind == "subst":
            flags = re.IGNORECASE if op.get("ignore_case") else 0
            pattern = re.compile(op["pattern"], flags)
            result = re.sub(r"\$(\d)", r"\\\1", op["result"])
            return pattern.sub(result, text)
        if kind == "run":
            target = op["map"]
            if target.startswith("map."):
                # dotted dependency reference: map.<alias>.stage.<name>
                alias = target.split(".")[1]
                deps = {
                    d.get("alias") or d["name"]: d["name"]
                    for d in self.tree.get("dependencies", [])
                    if isinstance(d, dict)
                }
                if alias not in deps:
                    raise ExecutionError(f"run {target!r}: unknown dependency alias")
                target = deps[alias]
            if self._loader is None:
                raise ExecutionError(f"run {op['map']!r}: no map loader configured")
            return self._loader(target).transliterate(text)
        if kind == "downcase":
            return text.lower()
        if kind == "upcase":
            return text.upper()
        if kind == "titlecase":
            return text.title()
        if kind == "compose":
            return unicodedata.normalize("NFC", text)
        if kind == "decompose":
            return unicodedata.normalize("NFD", text)
        what = op.get("what", kind)
        if self.on_unsupported == "skip":
            self.skipped_unsupported.append(str(what))
            return text
        raise ExecutionError(f"unsupported construct: {what!r}")
