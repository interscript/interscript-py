"""Interscript Python runtime — direct .imp DSL parsing + execution.

Public API (compatible with the pre-compiled module interface):
    map_exist(name), map_list(), load_map(name), transliterate(name, text)
"""
from __future__ import annotations

from pathlib import Path

from .engine import Engine, ExecutionError
from .parser import parse_file

__all__ = [
    "map_exist", "map_list", "load_map", "transliterate",
    "Engine", "ExecutionError", "parse_file",
]

_load_paths: list[Path] = []
_cache: dict[str, Engine] = {}


def add_load_path(path: str | Path) -> None:
    _load_paths.append(Path(path))


def _find_map(map_name: str) -> Path | None:
    for base in _load_paths:
        for ext in (".imp", ".isc"):
            candidate = base / f"{map_name}{ext}"
            if candidate.is_file():
                return candidate
    return None


def map_exist(map_name: str) -> bool:
    return _find_map(map_name) is not None


def map_list() -> list[str]:
    names = set()
    for base in _load_paths:
        if base.is_dir():
            for f in base.iterdir():
                if f.suffix in (".imp", ".isc"):
                    names.add(f.stem)
    return sorted(names)


def load_map(map_name: str) -> Engine:
    if map_name in _cache:
        return _cache[map_name]
    path = _find_map(map_name)
    if path is None:
        raise FileNotFoundError(
            f"map {map_name!r} not found in load paths "
            f"({', '.join(str(p) for p in _load_paths) or 'none configured'})"
        )
    tree = parse_file(path)
    engine = Engine(tree, loader=load_map)
    _cache[map_name] = engine
    return engine


def transliterate(map_name: str, text: str) -> str:
    return load_map(map_name).transliterate(text)
