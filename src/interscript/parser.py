"""Parser for the Interscript .imp map DSL.

Produces a plain tree:
  {"metadata": {k: str}, "tests": [(input, expected)],
   "stages": [ {"kind": "parallel", "subs": [(from, to)]}
             | {"kind": "subst", "pattern": str, "result": str}
             | {"kind": "run", "map": str}
             | {"kind": "downcase"|"upcase"|"titlecase"}
             | {"kind": "compose"|"decompose"} ]}
Unknown constructs parse into {"kind": "unsupported", "what": ...} and
are rejected loudly by the executor if reached.
"""

from __future__ import annotations

import codecs
import re
from pathlib import Path

_ADVANCED = ("any(", "boundary", "space", " + ")
_UNESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
_GUARD = re.compile(r"(before|after)\s*:\s*(.+?)(?=,\s*(?:before|after)\s*:|$)")


def _split_head(text: str) -> tuple[str, str]:
    """Split 'pattern, result' at the first top-level comma."""
    depth = 0
    in_str = False
    for i, ch in enumerate(text):
        if in_str:
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return text[:i].strip(), text[i + 1 :].strip()
    return text.strip(), ""


def _strip_comment(text: str) -> str:
    """Cut a trailing # comment (outside string literals)."""
    in_str = False
    for i, ch in enumerate(text):
        if ch == '"':
            in_str = not in_str
        elif ch == "#" and not in_str:
            return text[:i]
    return text


def parse_sub_args(text: str) -> dict | None:
    """Parse a sub line's argument list into pattern/result expressions
    plus optional before:/after: guard expressions."""
    text = _strip_comment(text)
    guards: dict[str, str] = {}
    tail = text
    for m in _GUARD.finditer(text):
        guards[m.group(1)] = m.group(2).strip().rstrip(",")
    if guards:
        first = min(m.start() for m in _GUARD.finditer(text))
        # back up to the comma preceding the first guard
        tail = text[:first].rstrip().rstrip(",")
    pattern, result = _split_head(tail)
    if not pattern:
        return None
    return {
        "pattern": pattern,
        "result": result,
        "before": guards.get("before"),
        "after": guards.get("after"),
    }

_LINE = re.compile(
    r"""^\s*
    (?:
      (?P<open>metadata|tests|stage|parallel|extend|reverse)\s*\{\s*(?P<comment>\#.*)?$ |
      (?P<close>\})\s*(?P<comment2>\#.*)?$ |
      (?P<key>[a-z_]+)\s*:\s*(?P<value>.*?)\s*(?P<comment3>\#.*)?$ |
      (?P<cmd>test|sub|subst|run|downcase|upcase|titlecase|compose|decompose
             |int_class|int_base|secryst|rababa|separate|unseparate|dependency)\b\s*(?P<args>.*)
    )""",
    re.VERBOSE,
)
_STR = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _split_args(text: str) -> list[str]:
    """Split comma-separated DSL args: bare words and quoted strings."""
    args: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        while pos < n and text[pos] in " \t":
            pos += 1
        if pos >= n or text[pos] == "#":
            break
        if text[pos] == '"':
            m = _STR.match(text, pos)
            if not m:
                raise ValueError(f"unterminated string in: {text!r}")
            raw = m.group(1)
            unescaped = _UNESCAPE.sub(lambda m2: chr(int(m2.group(1), 16)), raw)
            args.append(unescaped.replace('\\"', '"').replace("\\\\", "\\"))
            pos = m.end()
        elif text[pos] == "/":
            end = text.find("/", pos + 1)
            if end == -1:
                raise ValueError(f"unterminated regex in: {text!r}")
            args.append(text[pos : end + 1])
            pos = end + 1
        else:
            m = re.match(r"[^,#]+", text[pos:])
            word = m.group(0).strip() if m else text[pos:].strip()
            args.append(word)
            pos += m.end() if m else n - pos
        while pos < n and text[pos] in " \t":
            pos += 1
        if pos < n and text[pos] == ",":
            pos += 1
    return args


def _split_regex_arg(arg: str) -> tuple[str, str]:
    """'/pat/i' -> ('pat', 'i'); bare string -> itself, no flags."""
    if arg.startswith("/") and arg.endswith("/") and len(arg) >= 2:
        return arg[1:-1], ""
    if arg.startswith("/"):
        return arg[1 : arg.rfind("/")], arg[arg.rfind("/") + 1 :]
    return arg, ""


def parse_imp(text: str) -> dict:
    root: dict = {"metadata": {}, "tests": [], "stages": []}
    stack: list[dict] = [root]
    block_key: tuple[str, int] | None = None  # (key, indent) while inside `key: |`

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if block_key is not None:
            key, base_indent, free = block_key
            node = stack[-1]
            if line.strip() and len(line) - len(line.lstrip()) <= base_indent:
                block_key = None  # dedent ends the block scalar
            else:
                if not free:
                    node["pairs"][key] += "\n" + line.strip()
                continue
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"line {lineno}: cannot parse: {line.strip()[:80]!r}")

        if m.group("open"):
            kind = m.group("open")
            node: dict
            if kind == "metadata":
                node = {"_block": "metadata"}
            elif kind == "tests":
                node = {"_block": "tests"}
            elif kind in ("stage", "parallel"):
                node = {"kind": "stage" if kind == "stage" else "parallel", "subs": [], "children": []}
            else:
                node = {"_block": f"unsupported:{kind}"}
            stack.append(node)
            continue

        if m.group("close"):
            node = stack.pop()
            parent = stack[-1]
            if node.get("_block") == "metadata":
                parent["metadata"] = node.get("pairs", {})
            elif node.get("_block") == "tests":
                parent["tests"] = node.get("cases", [])
            elif node.get("kind") == "parallel" and parent.get("kind") == "stage":
                parent["children"].append({"kind": "parallel", "subs": node["subs"]})
            elif node.get("kind") == "stage":
                parent["stages"].append(node)
            continue

        if m.group("key") is not None:
            key, value = m.group("key"), m.group("value")
            node = stack[-1]
            if node.get("_block") == "metadata":
                if value.rstrip() == "|":
                    indent = len(raw) - len(raw.lstrip())
                    node.setdefault("pairs", {})[key] = ""
                    block_key = (key, indent, False)
                elif not value.strip():
                    indent = len(raw) - len(raw.lstrip())
                    block_key = (key, indent, True)  # valueless key: skip its block
                else:
                    node.setdefault("pairs", {})[key] = value.strip().strip('"')
            continue

        cmd, args_text = m.group("cmd"), m.group("args") or ""
        args = _split_args(args_text)
        node = stack[-1]

        if cmd == "dependency":
            if args:
                alias = None
                if len(args) >= 2 and args[1].startswith("as:"):
                    alias = args[1].split(":", 1)[1].strip()
                root.setdefault("dependencies", []).append(
                    {"name": args[0], "alias": alias}
                )
        elif cmd == "test" and len(args) >= 2:
            node.setdefault("cases", []).append((args[0], args[1]))
        elif cmd == "sub":
            parsed = parse_sub_args(args_text)
            if parsed is None:
                node["children"].append({"kind": "unsupported", "what": "sub-parse"})
            else:
                node.setdefault("subs", []).append(parsed)
        elif cmd == "subst" and args:
            pattern, flags = _split_regex_arg(args[0])
            result = args[1] if len(args) > 1 else ""
            entry = {"kind": "subst", "pattern": pattern, "result": result}
            if "i" in flags:
                entry["ignore_case"] = True
            if node.get("kind") == "stage":
                node["children"].append(entry)
        elif cmd == "run" and args:
            node["children"].append({"kind": "run", "map": args[0].strip('"')})
        elif cmd in ("downcase", "upcase", "titlecase", "compose", "decompose"):
            node["children"].append({"kind": cmd})
        else:
            node["children"].append({"kind": "unsupported", "what": cmd})

    return root


def parse_file(path: Path | str) -> dict:
    return parse_imp(Path(path).read_text(encoding="utf-8"))
