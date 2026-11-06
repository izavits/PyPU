from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from pypu.datasets import label_sar_propensity, label_scar
from pypu.models import SAREM, NaivePU
from pypu.models.sarem import _ConstantPropensity, _posterior, _resolve_features, _slope

REFERENCE = Path(__file__).with_name("data") / "sarem_reference.npz"

# Reference cases of data/make_sarem_reference.py -> the SAREM parameters that match them.
REFERENCE_CASES = {
    "experiments_scar": {},
    "experiments_sar": {},
    "propensity_subset": {
        "propensity_features": [0, 2],
        "convergence_window": 5,
        "tol": 1e-3,
        "refit": False,
    },
}


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def sar_data():
    """SAR labels whose propensity depends on feature 0 only."""
    X, y = make_classification(
        n_samples=3000,
        n_features=6,
        n_informative=4,
        weights=[0.7],
        class_sep=1.5,
        random_state=0,
    )
    s, propensity = label_sar_propensity(
        X,
        y,
        features=[0],
        label_frequency=0.4,
        strength=2.0,
        return_propensity=True,
        random_state=0,
    )
    return X, y, s, propensity


@pytest.mark.parametrize("case", list(REFERENCE_CASES))
def test_reproduces_the_original_implementation(reference, case):
    # The original computed its first sample weights in float32 (NumPy 1 casting float32
    # labels with a float64 scalar); with those it matches to 1e-15, with PyPU's float64
    # weights to about 1e-8.
    model = SAREM(**REFERENCE_CASES[case]).fit(reference["X_train"], reference[f"{case}_s"])
    assert model.n_iter_ == reference[f"{case}_iterations"]
    np.testing.assert_allclose(
        model.log_likelihoods_, reference[f"{case}_loglikelihoods"], rtol=0, atol=1e-7
    )
    X_test = reference["X_test"]
    np.testing.assert_allclose(
        model.predict_proba(X_test)[:, 1], reference[f"{case}_scores"], rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        model.predict_propensity(X_test), reference[f"{case}_propensity"], rtol=0, atol=1e-7
    )


def test_learns_the_propensity_from_the_features_that_drive_labeling(sar_data):
    X, y, s, propensity = sar_data
    model = SAREM(propensity_features=[0]).fit(X, s)
    positive = y == 1
    estimated = model.predict_propensity(X[positive])
    assert np.corrcoef(estimated, propensity[positive])[0, 1] > 0.95
    assert np.abs(estimated - propensity[positive]).mean() < 0.12
    naive = NaivePU().fit(X, s)
    scores, naive_scores = model.predict_proba(X)[:, 1], naive.predict_proba(X)[:, 1]
    assert roc_auc_score(y, scores) > roc_auc_score(y, naive_scores) + 0.1


def test_constant_propensity_without_features():
    X, y = make_classification(
        n_samples=3000, n_features=6, n_informative=4, weights=[0.7], class_sep=4, random_state=0
    )
    s = label_scar(y, label_frequency=0.4, random_state=0)
    model = SAREM(propensity_features=[]).fit(X, s)
    propensity = model.predict_propensity(X)
    assert np.ptp(propensity) == 0
    assert abs(propensity[0] - 0.4) < 0.06  # well-separated classes
    assert model.propensity_features_.size == 0


def test_constant_propensity_is_the_weighted_fraction_of_labeled_samples():
    s = np.array([1, 1, 0, 0, 0])
    weight = np.array([1.0, 1.0, 0.5, 0.5, 0.0])
    model = _ConstantPropensity().fit(np.zeros((5, 1)), s, sample_weight=weight)
    assert model.propensity_ == pytest.approx(2 / 3)
    np.testing.assert_allclose(model.predict_proba(np.zeros((2, 1))), [[1 / 3, 2 / 3]] * 2)


def test_posterior():
    f_x, e_x = np.array([0.5, 0.5, 0.8]), np.array([0.5, 0.5, 0.25])
    s = np.array([1, 0, 0])
    # Unlabeled: f (1 - e) / (1 - f e).
    np.testing.assert_allclose(_posterior(f_x, e_x, s), [1.0, 0.25 / 0.75, 0.6 / 0.8])


def test_slope_is_the_least_squares_slope():
    values = np.random.default_rng(0).normal(size=(4, 7))
    expected = [np.polyfit(np.arange(7), row, 1)[0] for row in values]
    np.testing.assert_allclose(_slope(values, axis=1), expected)


def test_log_likelihood_increases(sar_data):
    X, _, s, _ = sar_data
    model = SAREM(propensity_features=[0]).fit(X, s)
    assert model.log_likelihoods_.shape == (model.n_iter_ + 1,)
    assert model.log_likelihoods_[-1] > model.log_likelihoods_[0]


def test_warns_when_em_does_not_converge(pu_data):
    X, _, s = pu_data
    with pytest.warns(ConvergenceWarning, match="did not converge in 3 iterations"):
        model = SAREM(max_iter=3).fit(X, s)
    assert model.n_iter_ == 3


def test_refit_retrains_the_classifier_with_propensity_weights(pu_data):
    X, _, s = pu_data
    refit = SAREM().fit(X, s)
    no_refit = SAREM(refit=False).fit(X, s)
    assert refit.n_iter_ == no_refit.n_iter_
    np.testing.assert_allclose(refit.predict_propensity(X), no_refit.predict_propensity(X))
    assert not np.allclose(refit.predict_proba(X), no_refit.predict_proba(X))


def test_custom_estimators_are_cloned(pu_data):
    X, _, s = pu_data
    classifier, propensity = LogisticRegression(C=0.1), LogisticRegression(C=10.0)
    model = SAREM(estimator=classifier, propensity_estimator=propensity).fit(X, s)
    assert model.estimator_ is not classifier and model.estimator_.C == 0.1
    assert model.propensity_estimator_.C == 10.0
    assert not hasattr(classifier, "coef_")


@pytest.mark.parametrize(
    ("features", "expected"),
    [(None, [0, 1, 2, 3]), ([2, 0], [2, 0]), ([True, False, True, False], [0, 2]), ([], [])],
)
def test_resolve_features(features, expected):
    np.testing.assert_array_equal(_resolve_features(features, 4), expected)


@pytest.mark.parametrize(
    ("features", "match"),
    [
        ([0, 4], r"column indices in \[0, 3\]"),
        ([-1], r"column indices in \[0, 3\]"),
        ([True, False], "has 2 entries for 4 features"),
        ([0.5], "must hold column indices"),
        ([[0, 1]], "1-D"),
    ],
)
def test_rejects_invalid_propensity_features(features, match):
    with pytest.raises(ValueError, match=match):
        _resolve_features(features, 4)


@pytest.mark.parametrize(
    ("param", "estimator", "match"),
    [
        ("estimator", KNeighborsClassifier(), "must accept sample_weight"),
        ("propensity_estimator", KNeighborsClassifier(), "must accept sample_weight"),
        ("estimator", LinearSVC(), "must implement predict_proba"),
    ],
)
def test_rejects_unsuitable_estimators(pu_data, param, estimator, match):
    X, _, s = pu_data
    with pytest.raises(TypeError, match=f"{param} {match}"):
        SAREM(**{param: estimator}).fit(X, s)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"max_iter": 0}, "max_iter must be an int >= 1"),
        ({"max_iter": 2.5}, "max_iter must be an int >= 1"),
        ({"convergence_window": 1}, "convergence_window must be an int >= 2"),
        ({"tol": -1e-4}, "tol must be a finite number >= 0"),
        ({"tol": float("nan")}, "tol must be a finite number >= 0"),
    ],
)
def test_rejects_invalid_parameters(pu_data, params, match):
    X, _, s = pu_data
    with pytest.raises(ValueError, match=match):
        SAREM(**params).fit(X, s)
