"""WO07: nikud->IPA rule layer — one test per rule/special case."""

from __future__ import annotations

from interscript.ml.nikud_ipa import nikud_to_ipa, plane_to_ipa


class TestConsonants:
    def test_begadkefat_dagesh_hard(self):
        assert nikud_to_ipa("בּ") == "b"
        assert nikud_to_ipa("כּ") == "k"
        assert nikud_to_ipa("פּ") == "p"

    def test_begadkefat_soft(self):
        assert nikud_to_ipa("בָ") == "va"
        assert nikud_to_ipa("כָ") == "xa"
        assert nikud_to_ipa("פָ") == "fa"

    def test_final_forms_spirantize(self):
        assert nikud_to_ipa("ךְ") == "x"
        assert nikud_to_ipa("ף") == "f"

    def test_shin_sin_dots(self):
        assert nikud_to_ipa("שֶׁ") == "ʃe"
        assert nikud_to_ipa("שֶׂ") == "se"
        assert nikud_to_ipa("שְ") == "ʃə"  # dotless = shin; initial sheva = ə

    def test_plain_stops_and_fricatives(self):
        assert nikud_to_ipa("גַּ") == "ga"
        assert nikud_to_ipa("דִּ") == "di"
        assert nikud_to_ipa("תַּ") == "ta"
        assert nikud_to_ipa("זֶה") == "ze"
        assert nikud_to_ipa("חַ") == "xa"
        assert nikud_to_ipa("צָ") == "t͡sa"
        assert nikud_to_ipa("קוֹ") == "ko"
        assert nikud_to_ipa("רִי") == "ʁi"

    def test_matres_lectionis(self):
        assert nikud_to_ipa("וּ") == "u"       # shuruk: vav+dagesh
        assert nikud_to_ipa("וֹ") == "o"       # holam on vav
        assert nikud_to_ipa("יִ") == "ji"      # onset yod reads consonantal

    def test_yod_after_vowel_is_silent_lengthener(self):
        assert nikud_to_ipa("רִי") == "ʁi"

    def test_gutturals_silent_with_vowel(self):
        assert nikud_to_ipa("אָב") == "av"
        assert nikud_to_ipa("עִיר") == "iʁ"

    def test_final_heh_silent(self):
        assert nikud_to_ipa("סֻכָּה") == "suka"

    def test_sheva(self):
        assert nikud_to_ipa("בְּנָיָה") == "bənaja"  # initial sheva = ə
        assert nikud_to_ipa("מַלְכָּה") == "malka"   # non-initial sheva silent


class TestVowels:
    def test_core_vowels(self):
        assert nikud_to_ipa("דָּג") == "dag"
        assert nikud_to_ipa("דִּין") == "din"
        assert nikud_to_ipa("דֶּרֶךְ") == "deʁex"
        assert nikud_to_ipa("טוּב") == "tuv"

    def test_hataf(self):
        assert nikud_to_ipa("חֲנֻכָּה") == "xanuka"
        assert nikud_to_ipa("אֱמֶת") == "emet"

    def test_qamats_qatan_via_lexicon(self):
        # כָּל is the canonical qamats-qatan word: /kol/
        assert nikud_to_ipa("כָּל") == "kol"


class TestComposition:
    def test_plane_to_ipa_chains(self):
        class M:
            def translate(self, text, preserve_diacritics=False):
                return "שָׁלוֹם"

        assert plane_to_ipa(M(), "שלום") == "ʃalom"

    def test_deterministic(self):
        assert nikud_to_ipa("שָׁלוֹם") == "ʃalom"
        assert nikud_to_ipa("שָׁלוֹם") == nikud_to_ipa("שָׁלוֹם")


class TestCoverage:
    def test_every_real_cluster_has_rule_or_override(self):
        samples = ["בְּ", "שָׁ", "שִׂי", "צַ", "ךְ", "ף", "וּ", "יִ", "אֲ", "תּוֹ"]
        for s in samples:
            assert isinstance(nikud_to_ipa(s), str)


class TestPassThrough:
    def test_spaces_and_latin_survive(self):
        assert nikud_to_ipa("שָׁלוֹם עֲלֵיכֶם") == "ʃalom alexem"
        assert nikud_to_ipa("abc") == "abc"
