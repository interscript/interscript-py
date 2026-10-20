# CLAUDE.md — interscript-py

The official Python runtime: deterministic transliteration over the
ISC map corpus, parsing the map DSL directly. Project purpose and
direction: `interscript-models` README.

## Facts that bite

- The engine parses `.imp`/`.isc` directly — no precompiled map
  modules, no monorepo bootstrap in the runtime path.
- Release = push `v*` tag → PyPI trusted publishing. The publisher is
  currently registered against the archived `secryst/secryst-py` and
  must be re-registered by the owner to this repo before any release
  can exchange credentials (the 0.2.1 blocker).
- Maps that need vocalized input dispatch to a crystal through the
  optional phonological layer (`secryst` stdlib function); an engine
  without it cannot execute maps that declare a vocalization step.
- Measured coverage is stated against maps' own embedded tests — keep
  that table honest; document gaps in the suite, not in prose.

## Laws

- PRs only, never commit to main. Versions are owner decisions.
- No unconditional sibling-repo paths: dependencies resolve from the
  registry; local paths are conditional fallbacks only.
- Package immutability at runtime: no writes into the installed
  package.
