"""Gallery parity: the same conversion table the TypeScript example
gallery (interscript-ts/examples/) asserts. Same bytes from every
runtime - demonstrated, not claimed. Adding a case = one line."""
import pytest

import interscript

GALLERY = [
    ("Антон Олегович", "Anton Olehovych"),
    ("Соломія", "Solomiia"),
    ("Київ", "Kyiv"),
]


@pytest.mark.parametrize("input_text,expected", GALLERY)
def test_gallery_parity(input_text, expected):
    assert (
        interscript.transliterate("bgnpcgn-ukr-Cyrl-Latn-2019", input_text) == expected
    )
