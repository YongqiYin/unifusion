"""Reference context builders used for comparison.

Two controls that need no external model or query planner:

``dense_concat``
    Take the top-``budget`` chunks of the dense channel. This is the standard
    "retrieve and stuff" baseline and the natural ablation of the whole
    assembly stage.

``naive_fusion``
    Recall with the sparse channel and re-rank with the dense channel, i.e.
    concatenate both retrievers without any per-modality control. It shows what
    a merge alone buys before routing, quotas and redundancy penalties are
    applied.

Both follow the same calling convention as the controller: they see only score
rows and return corpus indices, so they can be evaluated with the identical
harness.
"""
from __future__ import annotations

import numpy as np

from .utils import topk_indices


def dense_concat(dense_scores: np.ndarray, budget: int = 10) -> list:
    """Top-``budget`` chunks of the dense channel, ranked by that channel."""
    if budget < 1:
        raise ValueError("budget must be >= 1")
    return [int(i) for i in topk_indices(np.asarray(dense_scores), budget)]


def naive_fusion(dense_scores: np.ndarray, sparse_scores: np.ndarray,
                 budget: int = 10, recall_k: int = 500) -> list:
    """Sparse recall of ``recall_k`` candidates, then dense re-ranking."""
    if budget < 1:
        raise ValueError("budget must be >= 1")
    dense_scores = np.asarray(dense_scores)
    sparse_scores = np.asarray(sparse_scores)
    if dense_scores.shape != sparse_scores.shape:
        raise ValueError("dense and sparse score rows must have the same shape")
    pool = topk_indices(sparse_scores, min(recall_k, len(sparse_scores)))
    if pool.size == 0:
        return []
    order = pool[np.argsort(-dense_scores[pool], kind="stable")]
    return [int(i) for i in order[:budget]]
