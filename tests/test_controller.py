import numpy as np
import pytest

from unifusion import ControllerConfig, QueryScores, UniFusionController

TABLE_FIRST = {"text": 0.15, "table": 0.80, "kg": 0.20}
FLAT = {"text": 0.50, "table": 0.50, "kg": 0.50}


def test_context_respects_budget_and_uniqueness(corpus, score_rows):
    dense, sparse = score_rows
    controller = UniFusionController(corpus, ControllerConfig(budget=10))
    selection = controller.select("alpha row", QueryScores(dense, sparse, FLAT))
    assert len(selection) == 10
    assert len(set(selection.indices)) == 10
    assert all(0 <= i < len(corpus) for i in selection.indices)
    assert selection.ids == [corpus.ids[i] for i in selection.indices]


def test_pool_cap_is_never_exceeded(corpus, score_rows):
    dense, sparse = score_rows
    config = ControllerConfig(budget=10, pool_cap=25)
    selection = UniFusionController(corpus, config).select(
        "alpha", QueryScores(dense, sparse, FLAT))
    assert selection.pool_size <= 25


def test_context_is_capped_by_the_pool(corpus, score_rows):
    """budget is an upper bound: a tiny pool yields a shorter context."""
    dense, sparse = score_rows
    config = ControllerConfig(budget=10, pool_cap=10,
                              per_modality_dense_k=2, per_modality_sparse_k=2)
    selection = UniFusionController(corpus, config).select(
        "alpha", QueryScores(dense, sparse, FLAT))
    assert len(selection) == selection.pool_size <= 10


def test_selection_is_deterministic(corpus, score_rows):
    dense, sparse = score_rows
    controller = UniFusionController(corpus, ControllerConfig(budget=8))
    first = controller.select("alpha row", QueryScores(dense, sparse, TABLE_FIRST))
    second = controller.select("alpha row", QueryScores(dense, sparse, TABLE_FIRST))
    assert first.indices == second.indices


def test_dense_anchor_is_always_selected(corpus, score_rows):
    dense, sparse = score_rows
    controller = UniFusionController(corpus, ControllerConfig(budget=6))
    selection = controller.select("alpha", QueryScores(dense, sparse, TABLE_FIRST))
    dense_top = int(np.argmax(dense))
    assert dense_top in selection.indices


def test_anchor_switch_changes_the_context(corpus):
    """Anchors pull in dense-channel leaders that the fused order would skip."""
    n = len(corpus)
    dense = np.zeros(n)
    sparse = np.zeros(n)
    table_block = corpus.modality_indices("table")[:4]
    dense[table_block] = 1.0                      # only 4 table chunks
    sparse[corpus.modality_indices("kg")[:8]] = 1.0  # fused order prefers kg
    probs = {"text": 0.10, "table": 0.20, "kg": 0.90}
    anchored = UniFusionController(corpus, ControllerConfig(budget=6)).select(
        "gamma", QueryScores(dense, sparse, probs)).indices
    plain = UniFusionController(corpus, ControllerConfig(budget=6, use_anchors=False)).select(
        "gamma", QueryScores(dense, sparse, probs)).indices
    assert anchored != plain
    assert all(int(i) in anchored for i in table_block)


def test_abi_switches_produce_valid_contexts(corpus, score_rows):
    dense, sparse = score_rows
    for config in (
        ControllerConfig(budget=6, use_redundancy=False),
        ControllerConfig(budget=6, use_entropy_adaptation=False),
        ControllerConfig(budget=6, use_soft_quotas=False),
        ControllerConfig(budget=6, use_anchors=False),
    ):
        selection = UniFusionController(corpus, config).select(
            "alpha", QueryScores(dense, sparse, FLAT))
        assert len(selection) == 6
        assert len(set(selection.indices)) == 6


def test_entropy_switch_selects_the_documented_weight_pair(corpus):
    """The weight gap between the two branches must equal the documented delta."""
    n = len(corpus)
    controller = UniFusionController(corpus, ControllerConfig(budget=5))
    cfg = controller.config
    pool = np.arange(n)
    dense_norm = np.linspace(1.0, 0.0, n)
    sparse_norm = np.linspace(0.0, 1.0, n)
    # a query token that appears nowhere -> overlap term is 0 everywhere
    flat = {"text": 0.5, "table": 0.5, "kg": 0.5}

    high, _, _ = controller._rerank(pool, dense_norm, sparse_norm, "zzz", flat, entropy=1.0)
    low, _, _ = controller._rerank(pool, dense_norm, sparse_norm, "zzz", flat, entropy=0.1)
    delta = ((cfg.w_dense_high_entropy - cfg.w_dense_low_entropy) * dense_norm
             + (cfg.w_sparse_high_entropy - cfg.w_sparse_low_entropy) * sparse_norm)
    assert np.allclose(high - low, delta)

    window = np.arange(n // 3)  # the fused order visibly changes
    assert not np.allclose(high[window], low[window])


def test_quotas_follow_the_modality_distribution(corpus):
    controller = UniFusionController(corpus, ControllerConfig(budget=10))
    quotas = controller._quotas({"text": 0.60, "table": 0.50, "kg": 0.20}, ["text"])
    assert quotas["active"] == ["text", "table"]
    assert quotas["total"] == {"text": 6, "table": 5}
    assert quotas["remaining"] == {"text": 5, "table": 5}


def test_constant_scores_do_not_crash(corpus):
    n = len(corpus)
    zeros = np.zeros(n)
    selection = UniFusionController(corpus, ControllerConfig(budget=7)).select(
        "alpha", QueryScores(zeros, zeros, FLAT))
    assert len(selection) == 7


def test_empty_modality_is_tolerated():
    from unifusion import Corpus, CorpusItem
    corpus = Corpus([CorpusItem(id=f"x{i}", text=f"passage {i}") for i in range(20)])
    zeros = np.zeros(len(corpus))
    selection = UniFusionController(corpus, ControllerConfig(budget=5)).select(
        "passage", QueryScores(zeros, zeros, {"text": 0.9, "table": 0.5, "kg": 0.4}))
    assert len(selection) == 5
    assert all(corpus[i].modality == "text" for i in selection.indices)


def test_config_validates_ranges():
    with pytest.raises(ValueError):
        ControllerConfig(budget=0)
    with pytest.raises(ValueError):
        ControllerConfig(budget=10, pool_cap=5)
    with pytest.raises(ValueError):
        ControllerConfig(tau_u=1.5)


def test_modality_mix_counts_selected_chunks(corpus, score_rows):
    dense, sparse = score_rows
    selection = UniFusionController(corpus, ControllerConfig(budget=10)).select(
        "alpha", QueryScores(dense, sparse, TABLE_FIRST))
    mix = selection.modality_mix(corpus)
    assert sum(mix.values()) == 10
