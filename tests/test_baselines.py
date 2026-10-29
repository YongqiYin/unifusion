import numpy as np
import pytest

from unifusion import dense_concat, naive_fusion


def test_dense_concat_returns_the_top_k_in_order():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert dense_concat(scores, budget=2) == [1, 3]
    assert dense_concat(scores, budget=4) == [1, 3, 2, 0]


def test_dense_concat_rejects_empty_budget():
    with pytest.raises(ValueError):
        dense_concat(np.zeros(3), budget=0)


def test_naive_fusion_recalls_with_sparse_then_ranks_with_dense():
    sparse = np.array([0.0, 0.9, 0.8, 0.0])
    dense = np.array([0.0, 0.2, 0.6, 0.9])
    # only indices 1 and 2 survive the sparse recall, dense picks 2 first
    assert naive_fusion(dense, sparse, budget=2, recall_k=2) == [2, 1]


def test_naive_fusion_respects_the_recall_window():
    sparse = np.array([0.0, 0.0, 0.9])
    dense = np.array([1.0, 0.9, 0.1])
    assert naive_fusion(dense, sparse, budget=2, recall_k=1) == [2]


def test_naive_fusion_rejects_mismatched_rows():
    with pytest.raises(ValueError):
        naive_fusion(np.zeros(3), np.zeros(4), budget=2)
