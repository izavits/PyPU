import contextlib
import sys

import numpy as np
import pytest
from sklearn.datasets import make_classification

# On macOS, LightGBM crashes when PyTorch was loaded before it (their OpenMP runtimes clash;
# see pypu.models._torch). Some test modules import torch when pytest collects them, so load
# LightGBM first.
if sys.platform == "darwin":
    with contextlib.suppress(ImportError, OSError):
        import lightgbm  # noqa: F401


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
