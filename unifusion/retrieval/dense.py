"""Dense-channel scoring on top of a sentence-embedding encoder.

The encoder is intentionally a thin wrapper: any bi-encoder can be plugged in,
and the controller only cares about the resulting score row.
"""
from __future__ import annotations

import numpy as np


class DenseEncoder:
    """CLS-pooled, L2-normalised embeddings from a HuggingFace bi-encoder.

    Parameters
    ----------
    model_name:
        Hub id or local path of the encoder. Defaults to the BGE English model
        used in the paper; the wrapper itself is architecture agnostic.
    pooling:
        ``"cls"`` (BGE-style) or ``"mean"``.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5",
                 device: str | None = None, batch_size: int = 64,
                 max_length: int = 512, pooling: str = "cls",
                 normalize: bool = True) -> None:
        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        self.normalize = normalize
        self._device = device
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------ lazy
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise ImportError(
                "DenseEncoder needs `torch` and `transformers`; "
                "install them with `pip install unifusion[dense]`."
            ) from exc
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(device).eval()

    # ---------------------------------------------------------------- public
    def encode(self, texts, progress: bool = False) -> np.ndarray:
        """Return an ``(len(texts), hidden)`` matrix of normalised embeddings."""
        self._ensure_loaded()
        import torch

        texts = list(texts)
        dimension = int(self._model.config.hidden_size)
        out = np.zeros((len(texts), dimension), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                encoded = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**encoded).last_hidden_state
                if self.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                if self.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out[start:start + len(batch)] = pooled.float().cpu().numpy()
                if progress and (start // self.batch_size) % 20 == 0:
                    print(f"[dense] {start + len(batch)}/{len(texts)}", flush=True)
        return out

    def scores(self, query_vectors: np.ndarray, corpus_vectors: np.ndarray) -> np.ndarray:
        """Inner-product score matrix ``(n_queries, n_corpus)``."""
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        corpus_vectors = np.asarray(corpus_vectors, dtype=np.float32)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors[None, :]
        return query_vectors @ corpus_vectors.T
