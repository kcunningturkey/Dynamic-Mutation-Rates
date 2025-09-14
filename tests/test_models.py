import numpy as np
import pytest
from src.models import (
    simulate_exponential_fixed,
    simulate_exponential_dynamic,
    simulate_logistic_fixed,
    simulate_logistic_dynamic,
    simulate_threshold_fixed,
    simulate_threshold_dynamic
)

@pytest.mark.parametrize("func", [
    simulate_exponential_fixed,
    simulate_exponential_dynamic,
    simulate_logistic_fixed,
    simulate_logistic_dynamic,
    simulate_threshold_fixed,
    simulate_threshold_dynamic
])
def test_model_output_shape_and_values(func):
    """Test that all models return valid proportions of strategy A."""
    N = 20
    generations = 50
    result = func(N=N, generations=generations)
    
    # Basic checks
    assert isinstance(result, list), f"{func.__name__} did not return a list"
    assert len(result) == generations, f"{func.__name__} returned wrong length"
    
    # Values must be between 0 and 1
    arr = np.array(result)
    assert np.all((arr >= 0) & (arr <= 1)), f"{func.__name__} produced invalid values"

