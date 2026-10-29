"""Lexical scoring: TF-IDF and Okapi BM25.

Both functions return a dense ``(n_queries, n_docs)`` float32 score matrix. That
is convenient for a few thousand queries; for web-scale corpora score the
candidates per query instead of materialising the whole matrix.
"""
from __future__ import annotations

import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenizer(text: str) -> list:
    return _TOKEN_RE.findall(str(text).lower())


def tfidf_scores(queries, docs, stop_words: str | None = "english",
                 ngram_range: tuple = (1, 2), max_features: int | None = 200_000,
                 min_df: int = 1, dtype=np.float32) -> np.ndarray:
    """Cosine-free TF-IDF inner products (query weights times document weights).

    ``min_df`` defaults to 1 because pruning rare terms is only safe once the
    corpus is large; raise it (e.g. 2) for corpora with millions of chunks.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError(
            "tfidf_scores needs scikit-learn; install it with "
            "`pip install unifusion[sparse]`."
        ) from exc
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=ngram_range,
                                 max_features=max_features, min_df=min_df)
    doc_matrix = vectorizer.fit_transform(list(docs))
    query_matrix = vectorizer.transform(list(queries))
    return (query_matrix @ doc_matrix.T).toarray().astype(dtype)


def bm25_scores(queries, docs, k1: float = 1.5, b: float = 0.75,
                chunk: int = 20_000, dtype=np.float32) -> np.ndarray:
    """Okapi BM25 over a shared vocabulary.

    The document term frequencies are saturated once
    (``tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``) and multiplied by
    the query's IDF-weighted term indicators, so the whole corpus can be scored
    in blocks.
    """
    try:
        import scipy.sparse as sp
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError(
            "bm25_scores needs scipy and scikit-learn; install them with "
            "`pip install unifusion[sparse]`."
        ) from exc

    vectorizer = CountVectorizer(tokenizer=_tokenizer, token_pattern=None,
                                 lowercase=False)
    doc_counts = vectorizer.fit_transform(list(docs)).astype(np.float32).tocsr()
    query_counts = vectorizer.transform(list(queries))

    doc_lengths = np.asarray(doc_counts.sum(axis=1)).ravel()
    avg_length = float(doc_lengths.mean()) if doc_lengths.size else 0.0
    if avg_length <= 0.0:
        return np.zeros((query_counts.shape[0], doc_counts.shape[0]), dtype=dtype)

    n_docs = doc_counts.shape[0]
    doc_freq = np.asarray((doc_counts > 0).sum(axis=0)).ravel()
    idf = np.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).astype(np.float32)

    scale = k1 * (1.0 - b + b * doc_lengths / avg_length)
    row_of_entry = np.repeat(np.arange(n_docs), np.diff(doc_counts.indptr))
    doc_counts.data = doc_counts.data * (k1 + 1.0) / (doc_counts.data + scale[row_of_entry])

    query_weights = (query_counts > 0).astype(np.float32) @ sp.diags(idf)

    out = np.zeros((query_counts.shape[0], n_docs), dtype=dtype)
    for start in range(0, n_docs, chunk):
        stop = min(n_docs, start + chunk)
        out[:, start:stop] = (query_weights @ doc_counts[start:stop].T).toarray()
    return out
