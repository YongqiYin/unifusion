import pytest

from unifusion.metrics import (aggregate_ranking, ap_at_k, eval_ranking,
                               exact_match, ndcg_at_k, precision_at_k,
                               recall_at_k, token_f1)


QRELS = {"a": 2, "b": 1, "c": 0}


def test_ndcg_is_one_for_the_ideal_ranking():
    assert ndcg_at_k(["a", "b", "c"], QRELS, 3) == pytest.approx(1.0)


def test_ndcg_penalises_a_swapped_order():
    graded = ndcg_at_k(["a", "b", "c"], QRELS, 3)
    swapped = ndcg_at_k(["b", "a", "c"], QRELS, 3)
    assert graded == 1.0
    assert swapped < graded
    assert ndcg_at_k(["c", "c"], QRELS, 2) == pytest.approx(0.0)


def test_unknown_chunk_ids_score_as_zero_gain():
    assert ndcg_at_k(["a", "zzz"], QRELS, 2) == ndcg_at_k(["a", "qqq"], QRELS, 2)
    assert ndcg_at_k(["a", "zzz"], QRELS, 2) < ndcg_at_k(["a", "b"], QRELS, 2)


def test_precision_counts_relevant_positions():
    assert precision_at_k(["a", "b", "c"], QRELS, 3) == pytest.approx(2 / 3)
    assert precision_at_k(["c", "c"], QRELS, 2) == pytest.approx(0.0)


def test_average_precision_divides_by_all_relevant_chunks():
    # hits at rank 1 and rank 3 -> (1/1 + 2/3) / 2
    assert ap_at_k(["a", "c", "b"], QRELS, 3) == pytest.approx((1.0 + 2 / 3) / 2)
    assert ap_at_k(["c"], QRELS, 1) == pytest.approx(0.0)


def test_recall_uses_the_relevant_set():
    assert recall_at_k(["a", "b", "c"], QRELS, 3) == pytest.approx(1.0)
    assert recall_at_k(["a"], QRELS, 1) == pytest.approx(0.5)


def test_ranking_suite_keys_and_aggregation():
    row = eval_ranking(["a", "b"], QRELS, ks=(1, 5), with_recall=True)
    assert set(row) == {"ndcg@1", "map@1", "p@1", "ndcg@5", "map@5", "p@5", "recall@10"}
    mean = aggregate_ranking([row, row])
    assert mean["ndcg@5"] == pytest.approx(row["ndcg@5"])


def test_qa_metrics_follow_squad_normalisation():
    assert exact_match("The Paris", "paris") == 1.0
    assert exact_match("Paris, France", "Paris") == 0.0
    assert token_f1("paris france", "paris france") == pytest.approx(1.0)
    assert token_f1("paris", "paris france") == pytest.approx(2 / 3)
    assert token_f1("", "") == 1.0
    assert token_f1("berlin", "paris") == 0.0
