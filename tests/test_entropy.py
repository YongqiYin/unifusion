import math

import numpy as np
import pytest

from unifusion.entropy import (MAX_ENTROPY, active_modalities, binary_entropy,
                               normalized_entropy)


def test_binary_entropy_bounds():
    assert binary_entropy(0.0) == pytest.approx(0.0, abs=1e-9)
    assert binary_entropy(1.0) == pytest.approx(0.0, abs=1e-9)
    assert binary_entropy(0.5) == pytest.approx(math.log(2.0))


def test_normalized_entropy_is_one_at_maximum_ambiguity():
    assert normalized_entropy([0.5, 0.5, 0.5]) == pytest.approx(1.0)
    assert MAX_ENTROPY == pytest.approx(3 * math.log(2.0))


def test_normalized_entropy_is_zero_for_extreme_probabilities():
    assert normalized_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)
    assert normalized_entropy([0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)


def test_normalized_entropy_is_monotone():
    confident = normalized_entropy([0.95, 0.10, 0.05])
    ambiguous = normalized_entropy([0.55, 0.50, 0.45])
    assert 0.0 < confident < ambiguous < 1.0


def test_entropy_accepts_mapping_and_array():
    assert normalized_entropy({"text": 0.9, "table": 0.2, "kg": 0.1}) == \
        pytest.approx(normalized_entropy(np.array([0.9, 0.2, 0.1])))


def test_active_set_uses_threshold():
    probs = {"text": 0.72, "table": 0.55, "kg": 0.12}
    assert active_modalities(probs, tau_a=0.40) == ["text", "table"]


def test_active_set_falls_back_to_argmax():
    probs = {"text": 0.10, "table": 0.30, "kg": 0.05}
    assert active_modalities(probs, tau_a=0.40) == ["table"]
