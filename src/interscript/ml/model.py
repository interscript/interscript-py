"""Model.load(zip) + translate(text): greedy KV decode with plain fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from interscript.ml.tokens import EOS_ID, PAD_ID, decode, encode
from interscript.ml.loader import load_manifest, verify_and_read


class Model:
    """A loaded, checksum-verified IMF v1 model.

    >>> model = Model.load("khm-latn-1.0.zip")
    >>> model.translate("ភាសា")
    """

    def __init__(self, zip_path: Path | str):
        self.zip_path = Path(zip_path)
        self.manifest = load_manifest(self.zip_path)
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        graphs = verify_and_read(self.zip_path)
        self._encoder = ort.InferenceSession(
            graphs["encoder.onnx"], options, providers=_providers()
        )
        decoder_name = (
            "decoder-kv.onnx"
            if self.manifest.decoder == "kv" and "decoder-kv.onnx" in graphs
            else "decoder.onnx"
        )
        self._kv_session = decoder_name == "decoder-kv.onnx"
        self._decoder = ort.InferenceSession(
            graphs[decoder_name], options, providers=_providers()
        )
        self._pasts = {
            meta.name: _zero_past(meta)
            for meta in self._decoder.get_inputs()
            if meta.name.startswith("past_")
        }
        self._output_names = [o.name for o in self._decoder.get_outputs()]

    @classmethod
    def load(cls, path_or_id: Path | str, index_url: str | None = None) -> "Model":
        """Accepts a zip path OR a model id from models.yaml (dynamic
        fetch: download -> verify -> cache)."""
        candidate = str(path_or_id)
        if candidate.endswith(".zip") or Path(candidate).exists():
            return cls(candidate)
        from interscript.ml.registry import resolve

        return cls(resolve(candidate, index_url))

    @property
    def id(self) -> str:
        return self.manifest.id

    def translate(self, text: str, max_len: int = 256) -> str:
        token_ids = self.generate(text, max_len=max_len)
        return decode(token_ids)

    def generate(self, text: str, max_len: int = 256) -> list[int]:
        ids = np.array([encode(text)], dtype=np.int64)
        if ids.shape[1] == 1:  # only the trailing EOS: empty input
            return []
        hidden = self._encoder.run(None, {"input_ids": ids})[0]
        if self._kv_session:
            return self._greedy_kv(hidden, max_len)
        return self._greedy_plain(hidden, max_len)

    def _greedy_kv(self, hidden, max_len: int) -> list[int]:
        pasts = dict(self._pasts)
        current = np.array([[PAD_ID]], dtype=np.int64)
        generated: list[int] = []
        for _ in range(max_len):
            outputs = self._decoder.run(
                None,
                {"input_ids": current, "encoder_hidden_states": hidden, **pasts},
            )
            results = dict(zip(self._output_names, outputs, strict=True))
            token = int(np.argmax(results["logits"][0, -1]))
            if token == EOS_ID:
                break
            generated.append(token)
            pasts = {
                name: results[name.replace("past_", "present_", 1)]
                for name in pasts
            }
            current = np.array([[token]], dtype=np.int64)
        return generated

    def _greedy_plain(self, hidden, max_len: int) -> list[int]:
        decoder_ids = np.array([[PAD_ID]], dtype=np.int64)
        generated: list[int] = []
        for _ in range(max_len):
            logits = self._decoder.run(
                None,
                {"input_ids": decoder_ids, "encoder_hidden_states": hidden},
            )[0]
            token = int(np.argmax(logits[0, -1]))
            if token == EOS_ID:
                break
            generated.append(token)
            decoder_ids = np.concatenate(
                [decoder_ids, np.array([[token]], dtype=np.int64)], axis=1
            )
        return generated


def _providers() -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    preferred = [p for p in ("CPUExecutionProvider",) if p in available]
    return preferred or available


def _zero_past(meta) -> object:
    shape = meta.shape  # [batch, heads, past_seq, d_kv], dynamic dims are str
    heads = shape[1] if isinstance(shape[1], int) else 4
    d_kv = shape[3] if isinstance(shape[3], int) else 8
    dtype = np.float16 if meta.type == "tensor(float16)" else np.float32
    return np.zeros((1, heads, 0, d_kv), dtype=dtype)
