"""interscript.ml — the neural layer: model index resolution and the
IMF v1 / plane zip runtimes (seq2seq byte models + plane-factorized).

Installed as an optional extra:  pip install interscript[ml].
Maps transliteration lives in the core package; this module resolves
model ids from the interscript-ml index and runs their ONNX graphs.

    from interscript.ml import Model, PlaneModel
    Model.load("khm-latn-1.0").translate("ភាសា")
"""

from interscript.ml.loader import Manifest, ModelFormatError
from interscript.ml.model import Model
from interscript.ml.plane import PlaneModel
from interscript.ml.registry import RegistryError, cache_dir, load_index, resolve
from interscript.ml.tokens import BYTE_OFFSET, EOS_ID, PAD_ID, UNK_ID, decode, encode

__all__ = [
    "BYTE_OFFSET",
    "EOS_ID",
    "Manifest",
    "Model",
    "ModelFormatError",
    "PAD_ID",
    "PlaneModel",
    "RegistryError",
    "UNK_ID",
    "cache_dir",
    "decode",
    "encode",
    "load_index",
    "resolve",
]
