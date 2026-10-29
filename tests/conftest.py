import numpy as np
import pytest

from unifusion import Corpus, CorpusItem


@pytest.fixture
def corpus():
    """30 chunks per modality, ids encode the modality."""
    items = []
    for i in range(30):
        items.append(CorpusItem(id=f"t{i:02d}", text=f"table row alpha item {i}",
                                modality="table"))
        items.append(CorpusItem(id=f"x{i:02d}", text=f"passage beta document {i}",
                                modality="text"))
        items.append(CorpusItem(id=f"g{i:02d}", text=f"entity gamma node {i}",
                                modality="kg"))
    return Corpus(items)


@pytest.fixture
def score_rows(corpus):
    """One dense-favouring row, one sparse-favouring row, both deterministic."""
    n = len(corpus)
    dense = np.linspace(1.0, 0.0, n)
    sparse = np.zeros(n)
    sparse[corpus.modality_indices("table")[:5]] = 1.0
    return dense, sparse
