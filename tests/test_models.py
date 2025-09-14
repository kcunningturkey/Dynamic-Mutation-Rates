import numpy as np
from src.models import simulate_exponential_fixed

def test_simulation_output_shape():
    history = simulate_exponential_fixed(N=50, generations=100)
    assert isinstance(history, list)
    assert len(history) == 100
    assert np.all(0 <= np.array(history)) and np.all(np.array(history) <= 1)
