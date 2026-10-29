"""Modality router (Eq. 2): q -> p(q) in [0, 1]^M.

A multi-label classifier rather than a single-label one, because a query can
legitimately need two modalities at once (e.g. a table lookup whose answer has
to be linked to an entity). The sigmoid outputs are consumed twice: directly as
the ``gamma * p_m`` prior in the fusion score, and through their entropy as the
switch that decides how much weight the sparse channel receives.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..types import MODALITIES


def _build_module(backbone: str, dropout: float):
    """Create the encoder + classification head (imported lazily)."""
    import torch.nn as nn
    from transformers import AutoModel

    class _RouterModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(backbone, add_pooling_layer=False)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.encoder.config.hidden_size, len(MODALITIES))

        def forward(self, input_ids, attention_mask):
            hidden = self.encoder(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state
            return self.classifier(self.dropout(hidden[:, 0]))

    return _RouterModule()


class ModalityRouter:
    """Multi-label modality classifier with an entropy-friendly interface."""

    def __init__(self, backbone: str = "microsoft/deberta-base",
                 dropout: float = 0.1, max_length: int = 256,
                 device: str | None = None) -> None:
        self.backbone = backbone
        self.dropout = dropout
        self.max_length = max_length
        self.device = device
        self._module = None
        self._tokenizer = None

    # ------------------------------------------------------------------ setup
    def _ensure_loaded(self) -> None:
        if self._module is not None:
            return
        import torch
        from transformers import AutoTokenizer
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        self._module = _build_module(self.backbone, self.dropout).to(self.device).eval()

    @property
    def module(self):
        """The underlying ``torch.nn.Module`` (built on first access)."""
        self._ensure_loaded()
        return self._module

    # ---------------------------------------------------------------- predict
    def predict_proba(self, texts, batch_size: int = 32) -> np.ndarray:
        """Return ``(len(texts), M)`` sigmoid probabilities in MODALITIES order."""
        import torch

        self._ensure_loaded()
        texts = list(texts)
        out = np.zeros((len(texts), len(MODALITIES)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                logits = self._module(encoded["input_ids"], encoded["attention_mask"])
                out[start:start + len(batch)] = torch.sigmoid(logits).float().cpu().numpy()
        return out

    # ------------------------------------------------------------ persistence
    def save(self, directory) -> None:
        import torch

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._ensure_loaded()
        torch.save(self._module.state_dict(), directory / "router.pt")
        (directory / "router.json").write_text(json.dumps({
            "backbone": self.backbone,
            "dropout": self.dropout,
            "max_length": self.max_length,
            "modalities": list(MODALITIES),
        }, indent=2))
        self._tokenizer.save_pretrained(directory / "tokenizer")

    @classmethod
    def load(cls, directory, device: str | None = None) -> "ModalityRouter":
        import torch
        from transformers import AutoTokenizer

        directory = Path(directory)
        meta = json.loads((directory / "router.json").read_text())
        router = cls(backbone=meta["backbone"], dropout=meta["dropout"],
                     max_length=meta.get("max_length", 256), device=device)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        router.device = device
        router._module = _build_module(meta["backbone"], meta["dropout"]).to(device)
        router._module.load_state_dict(torch.load(directory / "router.pt", map_location=device))
        router._module.eval()
        router._tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")
        return router
