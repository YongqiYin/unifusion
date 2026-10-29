"""Ranking metrics for graded relevance.

Conventions
-----------
* relevance grades ``L`` are non-negative integers (0, 1, 2, ...); ``L >= 1``
  marks a relevant chunk for the binary metrics.
* ``NDCG@k`` uses the exponential-free gain ``L`` and log2 discounting; the
  ideal ranking is the qrels truncated at ``k``.
* ``MAP@k`` divides by the total number of relevant chunks, not by ``min(k, R)``,
  so it cannot be inflated by a short relevance list.
"""
from __future__ import annotations

import numpy as np


def dcg_at_k(gains, k: int) -> float:
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(list(gains)[:k])))


def ndcg_at_k(ranked_ids, qrels: dict, k: int) -> float:
    """Normalised discounted cumulative gain over graded qrels."""
    gains = [float(qrels.get(d, 0)) for d in list(ranked_ids)[:k]]
    n_relevant = sum(1 for v in qrels.values() if v >= 1)
    ideal_len = min(k, n_relevant) if n_relevant else k
    ideal = sorted((float(v) for v in qrels.values()), reverse=True)[:ideal_len]
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0.0:
        return 0.0
    return dcg_at_k(gains, k) / idcg


def precision_at_k(ranked_ids, qrels: dict, k: int) -> float:
    top = list(ranked_ids)[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if qrels.get(d, 0) >= 1) / len(top)


def recall_at_k(ranked_ids, qrels: dict, k: int) -> float:
    n_relevant = sum(1 for v in qrels.values() if v >= 1)
    if n_relevant == 0:
        return 0.0
    return sum(1 for d in list(ranked_ids)[:k] if qrels.get(d, 0) >= 1) / n_relevant


def ap_at_k(ranked_ids, qrels: dict, k: int) -> float:
    """Average precision at ``k`` with denominator ``R`` (all relevant chunks)."""
    n_relevant = sum(1 for v in qrels.values() if v >= 1)
    if n_relevant == 0:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc in enumerate(list(ranked_ids)[:k], start=1):
        if qrels.get(doc, 0) >= 1:
            hits += 1
            total += hits / rank
    return total / n_relevant


def eval_ranking(ranked_ids, qrels: dict, ks=(1, 5), with_recall: bool = False) -> dict:
    """NDCG / MAP / P for every k in ``ks``."""
    out = {}
    for k in ks:
        out[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, qrels, k)
        out[f"map@{k}"] = ap_at_k(ranked_ids, qrels, k)
        out[f"p@{k}"] = precision_at_k(ranked_ids, qrels, k)
    if with_recall:
        out["recall@10"] = recall_at_k(ranked_ids, qrels, 10)
    return out


def aggregate(per_query: list) -> dict:
    """Arithmetic mean of every metric key over queries."""
    if not per_query:
        return {}
    return {key: float(np.mean([row[key] for row in per_query])) for key in per_query[0]}
