"""Tests of the defaults that BasePUClassifier gives a minimal subclass."""

import numpy as np
import pytest
from scipy.special import expit

from pypu.base import BasePUClassifier


class FirstFeatureModel(BasePUClassifier):
    """Minimal subclass: P(y=1|x) is the sigmoid of the first feature."""

    def _fit(self, X, s):
        self.fitted_ = True

    def _predict_proba(self, X):
        return expit(X[:, 0])


def test_cannot_instantiate_without_hooks():
    class Incomplete(BasePUClassifier):
        def _fit(self, X, s):
            pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_defaults(pu_data):
    X, _, s = pu_data
    model = FirstFeatureModel().fit(X, s)
    assert model.threshold_ == 0.5
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], expit(X[:, 0]))
    # The default decision function is the log-odds, so it is 0 where the probability is 0.5.
    np.testing.assert_allclose(model.decision_function(X), X[:, 0])
    np.testing.assert_array_equal(model.predict(X), (X[:, 0] >= 0).astype(int))


def test_float32_input_is_kept(pu_data):
    X, _, s = pu_data
    seen = {}

    class RecordingModel(FirstFeatureModel):
        def _fit(self, X, s):
            seen["dtype"] = X.dtype

    RecordingModel().fit(X.astype(np.float32), s)
    assert seen["dtype"] == np.float32


def test_score_uses_true_labels(pu_data):
    X, y, s = pu_data
    model = FirstFeatureModel().fit(X, s)
    assert model.score(X, y) == pytest.approx(np.mean(model.predict(X) == y))
