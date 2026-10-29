"""Data contracts between a retriever stack and the controller.

The controller is agnostic to where scores come from. A caller describes the
candidate pool once as a :class:`Corpus` and supplies, per query, one dense
score row, one sparse score row and one modality probability vector. Any
retriever that can produce those three artefacts can drive UniFusion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .utils import tokenize

#: Modality vocabulary used by the router and by the quota / fusion terms.
MODALITIES: tuple = ("text", "table", "kg")


@dataclass(frozen=True)
class CorpusItem:
    """One addressable unit of the candidate pool.

    ``modality`` is the only structural information the controller needs; how a
    chunk is assigned to a modality is a property of the corpus, not of the
    method.
    """

    id: str
    text: str
    modality: str = "text"

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(
                f"unknown modality {self.modality!r}; expected one of {MODALITIES}")


class Corpus:
    """Immutable candidate pool with per-modality index views."""

    def __init__(self, items: Iterable[CorpusItem]) -> None:
        items = list(items)
        if not items:
            raise ValueError("corpus is empty")
        ids = [it.id for it in items]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk ids in corpus")
        self.items: list = items
        self.ids: list = ids
        self.texts: list = [it.text for it in items]
        self.modalities: list = [it.modality for it in items]
        self.tokens: list = [tokenize(t) for t in self.texts]
        self.index_of: dict = {d: i for i, d in enumerate(ids)}
        self._by_modality: dict = {
            m: np.asarray([i for i, mm in enumerate(self.modalities) if mm == m],
                          dtype=np.int64)
            for m in MODALITIES
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> CorpusItem:
        return self.items[index]

    def modality_indices(self, modality: str) -> np.ndarray:
        """Row indices whose modality equals ``modality`` (may be empty)."""
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality {modality!r}")
        return self._by_modality[modality]

    def modality_counts(self) -> dict:
        return {m: int(v.size) for m, v in self._by_modality.items()}


@dataclass
class QueryScores:
    """Per-query retrieval evidence.

    Attributes
    ----------
    dense:
        Dense-channel score for every corpus row, shape ``(len(corpus),)``.
    sparse:
        Sparse-channel score for every corpus row, same length.
    probs:
        Modality probabilities from the router, either a length-3 array aligned
        with :data:`MODALITIES` or a ``{modality: probability}`` mapping.
    """

    dense: np.ndarray
    sparse: np.ndarray
    probs: "np.ndarray | dict"

    def as_prob_dict(self) -> dict:
        p = self.probs
        if isinstance(p, dict):
            missing = [m for m in MODALITIES if m not in p]
            if missing:
                raise ValueError(f"missing modality probabilities: {missing}")
            return {m: float(p[m]) for m in MODALITIES}
        arr = np.asarray(p, dtype=np.float64).ravel()
        if arr.size != len(MODALITIES):
            raise ValueError(
                f"expected {len(MODALITIES)} probabilities, got {arr.size}")
        return {m: float(v) for m, v in zip(MODALITIES, arr)}

    def validate(self, n_corpus: int) -> None:
        for name in ("dense", "sparse"):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (n_corpus,):
                raise ValueError(
                    f"{name} scores have shape {arr.shape}, expected ({n_corpus},)")


@dataclass
class ContextSelection:
    """Result of one call to :meth:`UniFusionController.select`."""

    indices: list
    ids: list
    entropy: float
    probs: dict
    active_modalities: list = field(default_factory=list)
    pool_size: int = 0

    def __len__(self) -> int:
        return len(self.indices)

    def modality_mix(self, corpus: Corpus) -> dict:
        out = {m: 0 for m in MODALITIES}
        for i in self.indices:
            out[corpus[i].modality] += 1
        return out
