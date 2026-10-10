from interscript.ml.model import first_alternates


def test_first_alternates_keeps_primary_choice_per_token():
    assert first_alternates("وِكَالَةُ/وِكَالَةُ اَلْأَرْصَادِ") == "وِكَالَةُ اَلْأَرْصَادِ"


def test_first_alternates_plain_text_untouched():
    assert first_alternates("مرحبا بالعالم") == "مرحبا بالعالم"


def test_first_alternates_edges():
    assert first_alternates("") == ""
    assert first_alternates("a/b c/d") == "a c"
    assert first_alternates("مَارْغِرِيتْ/مَارْغِرِيتْ/مَارْغِرِيتُ اَلثَّانِيَةُ") == "مَارْغِرِيتْ اَلثَّانِيَةُ"
