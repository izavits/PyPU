import numpy as np
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def pu_data():
    """Synthetic PU data: half of the positives labeled uniformly at random (SCAR).

    Returns ``(X, y, s)``: features, true labels and PU labels.
    """
    X, y = make_classification(
        n_samples=400, n_features=8, n_informative=5, weights=[0.7], random_state=0
    )
    rng = np.random.default_rng(0)
    s = ((y == 1) & (rng.random(y.shape[0]) < 0.5)).astype(int)
    return X, y, s
