import numpy as np
import pytest
from src.analysis import (
    entropy,
    run_with_confidence_intervals,
)

def test_entropy_valid_range():
    """Entropy should be between 0 and 1 for valid probabilities."""
    probs = np.linspace(0.01, 0.99, 50)
    H = entropy(probs)
    assert np.all(H >= 0), "Entropy should be non-negative"
    assert np.all(H <= np.log(2)), "Entropy should not exceed log(2) for binary distribution"

def test_entropy_zero_for_deterministic():
    """Entropy should be zero for p_A = 0 or 1 (deterministic states)."""
    assert entropy(0) == 0
    assert entropy(1) == 0

def test_run_with_confidence_intervals_shape():
    """Function should return mean and CI arrays of correct shape."""
    def dummy_sim(N, generations, p_A_init):
        return np.linspace(0, 1, generations)

    generations = 100
    mean, ci95 = run_with_confidence_intervals(
        dummy_sim, N=10, generations=generations, p_A_init=0.5
    )

    # Assertions
    assert mean.shape[0] == generations, "Mean output has wrong length"
    assert ci95.shape == mean.shape, "CI shape mismatch"
    assert np.all(ci95 >= 0), "CI values should be non-negative"

