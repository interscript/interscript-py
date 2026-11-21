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


def _effective_subs(subs: list[dict], language: str | None) -> list[dict]:
    """ISC-WF-1 W3 guard resolution for one parallel pass (§5.1/§5.2,
    ruby resolve_parallel_guards): the effective set = unguarded rules ∪
    guard-applicable rules; where an unguarded rule and a guard-
    applicable rule share the same from, the guard-applicable rule wins.
    Passes without any guard return the list untouched."""
    if not any(sub.get("when") for sub in subs):
        return subs
    applicable = [
        sub
        for sub in subs
        if not sub.get("when") or (language is not None and language in sub["when"])
    ]
    guarded_froms = {sub["pattern"] for sub in applicable if sub.get("when")}
    return [
        sub
        for sub in applicable
        if sub.get("when") or sub["pattern"] not in guarded_froms
    ]


def _select_language(tree: dict, language: str | None) -> str | None:
    """ISC-WF-1 W3 language selection (ruby Interpreter.select_language):
    a selection is honored only when the map declares it; anything else
    (no selection, undeclared selection) leaves every guard inert."""
    if language is None:
        return None
    declared = tree.get("languages")
    if not declared or language not in declared:
        return None
    return language


class Engine:
    def __init__(self, tree: dict, loader=None, on_unsupported: str = "raise") -> None:
        self.tree = tree
        self.metadata = tree.get("metadata", {})
        self._loader = loader
        self.on_unsupported = on_unsupported
        self.skipped_unsupported: list[str] = []
        # Compiled-op caches, keyed by id(op): op dicts are owned by
        # self.tree, which is never mutated after construction, so ids
        # are stable and unique for the Engine's lifetime. The former
        # single-slot cache recompiled on every call for any map with
        # more than one parallel op (slot thrash) and subst patterns
        # were recompiled on every call. Parallel entries are keyed by
        # (id(op), language) — the W3 guard resolution is per language.
        self._parallel_cache: dict[tuple[int, str | None], tuple[re.Pattern[str], dict[str, str]]] = {}
        self._subst_cache: dict[int, re.Pattern[str]] = {}

    def transliterate(self, text: str, language: str | None = None) -> str:
        language = _select_language(self.tree, language)
        for stage in self.tree.get("stages", []):
            text = self._run_stage(stage, text, language)
        return text

    def _run_stage(self, stage: dict, text: str, language: str | None = None) -> str:
        for child in stage.get("children", []):
            text = self._run_op(child, text, language)
        return text

    def _run_op(self, op: dict, text: str, language: str | None = None) -> str:
        kind = op.get("kind")
        if kind == "parallel":
            entry = self._parallel_cache.get((id(op), language))
            if entry is None:
                entry = _compile_parallel(_effective_subs(op["subs"], language))
                self._parallel_cache[(id(op), language)] = entry
            pattern, results = entry
            return pattern.sub(lambda m, results=results: results[m.lastgroup], text)
        if kind == "subst":
            when = op.get("when")
            if when and (language is None or language not in when):
                # W3: an inert guarded rule is skipped (§5.1).
                return text
            flags = re.IGNORECASE if op.get("ignore_case") else 0
            pattern = self._subst_cache.get(id(op))
            if pattern is None:
                pattern = re.compile(op["pattern"], flags)
                self._subst_cache[id(op)] = pattern
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
            dep = self._loader(target)
            stage_name = op.get("stage")
            if not stage_name:
                return dep.transliterate(text, language=language)
            out = text
            for st in dep.tree.get("stages", []):
                if st.get("name") == stage_name:
                    out = dep._run_stage(st, out, language)
            return out
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
