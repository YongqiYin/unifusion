import numpy as np
import pytest

pytest.importorskip("sklearn", reason="sparse scorers need scikit-learn")

from unifusion.retrieval import bm25_scores, tfidf_scores  # noqa: E402

DOCS = [
    "the stadium opened in 1990 with a capacity of 80000 seats",
    "a bridge spanning the canal was rebuilt after the flood",
    "the festival moved to the district rail terminal",
]
QUERIES = ["stadium capacity", "canal flood"]


def test_shapes_match_queries_by_docs():
    assert bm25_scores(QUERIES, DOCS).shape == (2, 3)
    assert tfidf_scores(QUERIES, DOCS).shape == (2, 3)


def test_matching_document_wins():
    for scores in (bm25_scores(QUERIES, DOCS), tfidf_scores(QUERIES, DOCS)):
        top = np.argmax(scores, axis=1)
        assert top.tolist() == [0, 1]


def test_unmatched_terms_score_zero():
    scores = bm25_scores(["zebra"], DOCS)
    assert np.allclose(scores, 0.0)


def test_scores_are_finite():
    for scores in (bm25_scores(QUERIES, DOCS), tfidf_scores(QUERIES, DOCS)):
        assert np.isfinite(scores).all()
