"""Predictive-entropy signals that condition the fusion weights (Eq. 3)."""
from __future__ import annotations

import math

import numpy as np

from .types import MODALITIES

#: Upper bound of the normalised entropy: every modality at p = 0.5.
MAX_ENTROPY = len(MODALITIES) * math.log(2.0)


def binary_entropy(p: float) -> float:
    """Entropy of one Bernoulli variable in nats, clipped away from 0 and 1."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def normalized_entropy(probs) -> float:
    """u(q) = 1 / (M ln 2) * sum_m H_b(p_m(q)), in [0, 1].

    ``probs`` is a length-3 array or a ``{modality: probability}`` mapping; both
    are read in :data:`MODALITIES` order.
    """
    if isinstance(probs, dict):
        values = [float(probs.get(m, 0.0)) for m in MODALITIES]
    else:
        values = np.asarray(probs, dtype=np.float64).ravel().tolist()
        if len(values) != len(MODALITIES):
            raise ValueError(
                f"expected {len(MODALITIES)} probabilities, got {len(values)}")
    return float(sum(binary_entropy(p) for p in values) / MAX_ENTROPY)


def active_modalities(probs: dict, tau_a: float = 0.40) -> list:
    """A(q) = {m : p_m(q) >= tau_a}; falls back to the arg-max modality.

    The active set sizes the quota denominator in Eq. (6), so it is never
    allowed to be empty.
    """
    active = [m for m in MODALITIES if float(probs.get(m, 0.0)) >= tau_a]
    if not active:
        active = [max(MODALITIES, key=lambda m: float(probs.get(m, 0.0)))]
    return active
