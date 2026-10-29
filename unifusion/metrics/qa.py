"""Answer-level metrics for extractive question answering.

``normalize_answer`` follows the SQuAD convention (lowercase, strip articles
and punctuation, collapse whitespace) so that formatting differences do not
dominate the score.
"""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text) -> str:
    text = str(text).lower()
    text = _ARTICLES.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    return " ".join(text.split())


def exact_match(prediction, gold) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction, gold) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_suite(prediction, gold) -> dict:
    return {"f1": token_f1(prediction, gold), "em": exact_match(prediction, gold)}


def aggregate(per_query: list) -> dict:
    """Arithmetic mean of every metric key over queries."""
    if not per_query:
        return {}
    return {key: float(np.mean([row[key] for row in per_query])) for key in per_query[0]}
