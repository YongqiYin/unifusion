import numpy as np
import pytest

from unifusion import Corpus, CorpusItem, QueryScores
from unifusion.entropy import normalized_entropy


def test_unknown_modality_rejected():
    with pytest.raises(ValueError):
        CorpusItem(id="a", text="t", modality="image")


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        Corpus([CorpusItem(id="a", text="1"), CorpusItem(id="a", text="2")])


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        Corpus([])


def test_modality_index_matches_ids(corpus):
    assert corpus.modality_counts() == {"text": 30, "table": 30, "kg": 30}
    table_rows = corpus.modality_indices("table")
    assert all(corpus.ids[i].startswith("t") for i in table_rows)
    assert corpus.modality_indices("table").dtype == np.int64


def test_query_scores_accepts_array_and_mapping():
    from_array = QueryScores(dense=np.zeros(3), sparse=np.zeros(3),
                             probs=np.array([0.1, 0.2, 0.7]))
    from_dict = QueryScores(dense=np.zeros(3), sparse=np.zeros(3),
                            probs={"text": 0.1, "table": 0.2, "kg": 0.7})
    assert from_array.as_prob_dict() == from_dict.as_prob_dict()
    assert pytest.approx(normalized_entropy(from_array.probs)) == \
        normalized_entropy(from_dict.probs)


def test_query_scores_rejects_short_probability_vector():
    scores = QueryScores(dense=np.zeros(3), sparse=np.zeros(3), probs=np.array([0.5]))
    with pytest.raises(ValueError):
        scores.as_prob_dict()


def test_query_scores_rejects_wrong_row_length():
    scores = QueryScores(dense=np.zeros(2), sparse=np.zeros(3), probs=np.zeros(3))
    with pytest.raises(ValueError):
        scores.validate(3)
