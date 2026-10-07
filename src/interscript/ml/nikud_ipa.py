"""nikud->IPA: deterministic Hebrew phonemization rules (WO07).

Standard Israeli reading over letter x nikud-cluster pairs. Cluster
order is the corpus convention (as written); shin/sin dots carry their
own meaning. Documented limitations: no stress marks, no spoken-norm
variation (that is the ReNikud audio-supervised territory), no teamim.
Qamats-qatan is lexicon-resolved; overrides beat rules.
"""

from __future__ import annotations

NIKUD = set("ְֱֲֳִֵֶַָֹֺֻּֽֿׁׂ")
DAGESH = "ּ"

_LETTERS = {
    "א": "", "ב": "v", "בּ": "b", "ג": "g", "ד": "d", "ה": "h",
    "ו": "v", "ז": "z", "ח": "x", "ט": "t", "י": "j",
    "כ": "x", "כּ": "k", "ך": "x", "ל": "l", "מ": "m", "ם": "m",
    "נ": "n", "ן": "n", "ס": "s", "ע": "", "פ": "f", "פּ": "p",
    "ף": "f", "צ": "t͡s", "ץ": "t͡s", "ק": "k", "ר": "ʁ", "שׁ": "ʃ",
    "שׂ": "s", "ש": "ʃ", "ת": "t", "תּ": "t",
}
_VOWELS = {
    "ַ": "a", "ָ": "a", "ֲ": "a", "ֱ": "e", "ֳ": "o", "ִ": "i",
    "ֵ": "e", "ֶ": "e", "ֹ": "o", "ֻ": "u", "ְ": "",
}
_LEXICON = {
    "כָּל": "kol",   # qamats qatan
    "נָא": "na",
}
_FINAL_FORMS = set("ךםןףץ")
_VOWEL_IPA = set("aeiouə")


def _parse(text: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    for ch in text:
        if ch in NIKUD and ch not in "ׁׂ":
            if units:
                units[-1] = (units[-1][0], units[-1][1] + ch)
            else:
                units.append(("\x00", ch))
            continue
        if units and units[-1][0] in ("ש", "שׁ", "שׂ") and ch in "ׁׂ":
            units[-1] = ("ש" + ch, units[-1][1])
            continue
        units.append((ch, ""))
    return units


def nikud_to_ipa(text: str) -> str:
    if text in _LEXICON:
        return _LEXICON[text]
    out: list[str] = []
    units = _parse(text)
    for i, (letter, cluster) in enumerate(units):
        last = i == len(units) - 1
        vowels = "".join(_VOWELS.get(c, "") for c in cluster if c in _VOWELS)

        if letter == "\x00":  # leading marks: emit the vowel only
            out.append(vowels)
            continue

        if letter == "ו" and cluster:
            if DAGESH in cluster:  # shuruk
                out.append("u")
                continue
            if "ֹ" in cluster:  # holam on vav
                out.append("o")
                continue

        if letter == "י":
            prev = out[-1] if out else ""
            if "ִ" in cluster:
                out.append("i" if prev and prev[-1] in _VOWEL_IPA else "ji")
                continue
            if not cluster and prev and prev[-1] in _VOWEL_IPA:
                continue  # bare yod after a vowel: silent lengthener
            if "ֵ" in cluster:
                out.append(vowels)  # yod as tsere-mater
                continue

        if letter in ("א", "ע"):
            # gutturals carry ʔ when vocalized (Israeli/TTS convention),
            # fully silent when bare
            out.append("ʔ" + vowels if vowels else "")
            continue

        if letter == "ה" and last:
            out.append(vowels)  # final heh is silent
            continue

        # begadkefat: the dagesh spelling is the hard-stop key
        if DAGESH in cluster:
            base = _LETTERS.get(letter + DAGESH, _LETTERS.get(letter, ""))
        else:
            base = _LETTERS.get(letter, "")
        if not base and letter not in _LETTERS and not cluster:
            out.append(letter)  # non-Hebrew: pass through (spaces, punct)
            continue
        if not vowels and "ְ" in cluster:
            # sheva: word-initial onset = ə, otherwise silent
            if not out and letter not in _FINAL_FORMS:
                vowels = "ə"
        out.append(base + vowels)
    return "".join(out)


def plane_to_ipa(model, text: str) -> str:
    """Chain a plane model's nikud restoration into the IPA rules."""
    return nikud_to_ipa(model.translate(text))
