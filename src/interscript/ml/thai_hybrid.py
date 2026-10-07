"""Thai lexicon fast-path + neural fallback (WO03).

The FastThaiG2P response: dictionary hits decode at dict speed,
out-of-vocabulary runs go to the neural model as whole spans. Maximal-
munch longest-match segmentation; spaces/punct pass through untouched.
"""

from __future__ import annotations

_THAI = "฀-๿"


class Lexicon(dict):
    """word -> IPA; longest key wins via sorted lookup."""


def _longest_at(text: str, i: int, keys: list[str], max_len: int) -> str | None:
    for n in range(min(max_len, len(text) - i), 0, -1):
        w = text[i:i + n]
        if w in keys:
            return w
    return None


def segment(text: str, lex: Lexicon) -> list[tuple[str, str | None]]:
    """-> [(chunk, ipa-or-None)]; None = OOV run for the model."""
    keys = lex.keys()
    max_len = max((len(k) for k in keys), default=0)
    out: list[tuple[str, str | None]] = []
    oov: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if not _is_thai(ch):
            if oov:
                out.append(("".join(oov), None))
                oov = []
            out.append((ch, ch))
            i += 1
            continue
        w = _longest_at(text, i, keys, max_len)
        if w:
            if oov:
                out.append(("".join(oov), None))
                oov = []
            out.append((w, lex[w]))
            i += len(w)
        else:
            oov.append(ch)
            i += 1
    if oov:
        out.append(("".join(oov), None))
    return out


def _is_thai(ch: str) -> bool:
    return "฀" <= ch <= "๿"


def hybrid_translate(text: str, lex: Lexicon, model) -> str:
    """Dict hits splice verbatim; each maximal OOV run is one model call."""
    parts: list[str] = []
    for chunk, ipa in segment(text, lex):
        if ipa is None:
            parts.append(model.translate(chunk))  # Thai OOV run
        elif not _is_thai(chunk[0]):
            parts.append(chunk)  # spaces/punct pass through
        else:
            parts.append(ipa)  # dictionary hit
    return "".join(parts)


def build_lexicon_from_kaikki(entries, min_head_len: int = 1) -> Lexicon:
    """kaikki Thai entries (dicts with 'word' + 'sounds') -> Lexicon."""
    counts: dict[str, dict[str, int]] = {}
    for e in entries:
        word = e.get("word") or ""
        if len(word) < min_head_len:
            continue
        for s in e.get("sounds") or []:
            ipa = s.get("ipa")
            if not ipa:
                continue
            ipa = ipa.strip("/")
            if ipa:
                counts.setdefault(word, {ipa: 0})
                counts[word][ipa] = counts[word].get(ipa, 0) + 1
    lex = Lexicon()
    for word, ipas in counts.items():
        lex[word] = max(ipas.items(), key=lambda kv: kv[1])[0]
    return lex
