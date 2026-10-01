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

from .expr import expr_is_par_safe, expr_lookbehind, expr_max_length, expr_neg_lookbehind, expr_to_literal, expr_to_regex, is_plain_string


class ExecutionError(ValueError):
    """The map uses a construct this engine does not implement yet."""


def _compile_parallel(subs: list[dict]) -> tuple[re.Pattern[str], dict[str, str]]:
    """Compile one parallel group: longest-pattern-first alternation
    with a named group per sub; lookaround guards for before:/after:.

    An all-plain group compiles to a replace tree in Ruby, where a
    later duplicate from overwrites an earlier one; a guarded group
    takes the megaregexp, where the earliest equal-key rule wins."""
    guards = ("before", "after", "not_before", "not_after")
    all_plain = not any(sub.get(g) for sub in subs for g in guards) and all(
        sub.get("result") is not None
        and expr_is_par_safe(sub["pattern"])
        and expr_is_par_safe(sub["result"])
        for sub in subs
    )
    drop: set[int] = set()
    if all_plain:
        last: dict[str, int] = {}
        for i, sub in enumerate(subs):
            if is_plain_string(sub["pattern"]):
                last[expr_to_literal(sub["pattern"])] = i
        for i, sub in enumerate(subs):
            if is_plain_string(sub["pattern"]) and last[expr_to_literal(sub["pattern"])] != i:
                drop.add(i)
    indexed = []
    for i, sub in enumerate(subs):
        if i in drop:
            continue
        pat = expr_to_regex(sub["pattern"])
        full = pat
        if sub.get("before"):
            full = expr_lookbehind(sub["before"]) + full
        if sub.get("not_before"):
            full = expr_neg_lookbehind(sub["not_before"]) + full
        if sub.get("not_after"):
            full = full + "(?!" + expr_to_regex(sub["not_after"]) + ")"
        if sub.get("after"):
            full = full + "(?=" + expr_to_regex(sub["after"]) + ")"
        key = expr_max_length(sub["pattern"])
        for guard in ("before", "after", "not_before", "not_after"):
            if sub.get(guard):
                key += expr_max_length(sub[guard])
        key += sub.get("priority", 0)
        indexed.append((key, full, f"s{i}"))
    indexed.sort(key=lambda t: -t[0])
    combined = "|".join(f"(?P<{name}>{full})" for _, full, name in indexed)
    pattern = re.compile(combined) if indexed else re.compile(r"(?!)")

    results = {f"s{i}": expr_to_literal(sub["result"]) for i, sub in enumerate(subs)}
    return pattern, results


_CASE_FNS = {
    "upcase": str.upper,
    "downcase": str.lower,
    "title_case": str.title,
    "swapcase": str.swapcase,
    "strip": str.strip,
    "reverse": lambda s: s[::-1],
}


class Engine:
    def __init__(self, tree: dict, loader=None, on_unsupported: str = "raise") -> None:
        self.tree = tree
        self.metadata = tree.get("metadata", {})
        self._loader = loader
        self.on_unsupported = on_unsupported
        self.skipped_unsupported: list[str] = []
        self._compiled: re.Pattern[str] | None = None
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

    def _run_op(self, op: dict, text: str) -> str:
        kind = op.get("kind")
        if kind == "parallel":
            if self._compiled is None or self._compiled_source != id(op):
                pattern, results = _compile_parallel(op["subs"])
                self._compiled = pattern
                self._group_results = results
                self._compiled_source = id(op)
            return self._compiled.sub(lambda m: self._group_results[m.lastgroup], text)
        if kind == "subst":
            flags = re.IGNORECASE if op.get("ignore_case") else 0
            pattern = re.compile(op["pattern"], flags)
            case = op.get("case")
            if case:
                fn = _CASE_FNS[case]
                return pattern.sub(lambda m: fn(m.group(0)), text)
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
        if kind == "separate":
            separator = op.get("separator", " ")
            return separator.join(text)
        if kind == "title_case":
            # Ruby parity: upcase the first character of every line and
            # of every segment following the word separator.
            sep = op.get("word_separator", " ")
            out = "\n".join(
                line[:1].upper() + line[1:] for line in text.split("\n")
            )
            if sep != "":
                out = re.sub(
                    f"(?<={re.escape(sep)})(.)",
                    lambda m: m.group(1).upper(),
                    out,
                )
            return out
        if kind == "titlecase":
            return text.title()
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
