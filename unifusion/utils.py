"""Text and array helpers shared across the package."""
from __future__ import annotations

import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set:
    """Lowercase alphanumeric tokens.

    Overlap and redundancy signals are computed on this tokenisation, so it is
    kept deliberately simple and deterministic.
    """
    return set(_TOKEN_RE.findall(str(text).lower()))


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two token sets, 0.0 when either side is empty."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / max(1, len(a | b))


def minmax(values: np.ndarray) -> np.ndarray:
    """Min-max scale a score vector to [0, 1]; constant input maps to zeros.

    The input dtype is preserved: score rows are usually float32 and upcasting
    them would change tie-breaking between near-equal candidates.
    """
    values = np.asarray(values)
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, sorted descending (stable on ties)."""
    scores = np.asarray(scores)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if len(scores) <= k:
        return np.argsort(-scores, kind="stable")
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]
