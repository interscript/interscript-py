import os
from pathlib import Path

import pytest

import interscript

MAPS = Path(os.environ.get("INTERSCRIPT_MAPS_PATH", Path(__file__).parent.parent.parent / "maps" / "maps"))

def assert_transliteration(map, input, output):
    interscript.load_map(map)
    result = interscript.transliterate(map, input)
    assert result == output

@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_can_transliterate():
    assert_transliteration('bgnpcgn-ukr-Cyrl-Latn-2019', 'Антон', 'Anton')

@pytest.mark.skipif(not MAPS.is_dir(), reason="interscript maps repo not present")
def test_can_transliterate_maps_requiring_libraries():
    assert_transliteration('bgnpcgn-deu-Latn-Latn-2000', 'Tschüß!', 'Tschueß!')
