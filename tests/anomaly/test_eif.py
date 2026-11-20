import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks

from pypu.anomaly import ExtendedIsolationForest
from pypu.anomaly._eif import _c_factor

REFERENCE = Path(__file__).with_name("data") / "eif_reference.npz"

# Cases of data/make_eif_reference.py -> (seed, the matching parameters).
EXACT_CASES = {
    "level1": (0, {"n_estimators": 20, "max_samples": 128, "extension_level": 1}),
    "full": (1, {"n_estimators": 20, "max_samples": 300, "extension_level": 3}),
    "axis_parallel_depth4": (
        2,
        {"n_estimators": 10, "max_samples": 64, "extension_level": 0, "max_depth": 4},
    ),
}


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture
def X_with_outlier():
    X = np.random.default_rng(0).normal(size=(400, 3))
    X[17] = 6.0
    return X


def anomaly_scores(detector, X):
    return -detector.score_samples(X)


@pytest.mark.parametrize("case", list(EXACT_CASES))
def test_reproduces_eif_old_from_the_same_seed(reference, case):
    seed, params = EXACT_CASES[case]
    detector = ExtendedIsolationForest(random_state=seed, **params).fit(reference["X"])
    np.testing.assert_array_equal(
        anomaly_scores(detector, reference["X"]), reference[f"{case}_scores"]
    )


def test_agrees_with_the_cpp_eif_of_the_experiments(reference):
    # The C++ eif draws its random numbers differently: compare with its spread over seeds.
    X, cpp = reference["X"], reference["cpp_scores"]
    scores = anomaly_scores(ExtendedIsolationForest(random_state=0).fit(X), X)
    mean = cpp.mean(axis=0)
    # How well one C++ run agrees with the mean of the others.
    others = [spearmanr(cpp[i], np.delete(cpp, i, axis=0).mean(axis=0))[0] for i in range(10)]
    assert spearmanr(scores, mean)[0] >= min(others) - 0.01
    run_means = cpp.mean(axis=1)
    assert run_means.min() - 0.005 <= scores.mean() <= run_means.max() + 0.005


def test_defaults_are_the_experiments_settings(X_with_outlier):
    detector = ExtendedIsolationForest(random_state=0).fit(X_with_outlier)
    assert (detector.n_estimators, detector.extension_level) == (100, 1)
    assert (detector.max_samples_, detector.max_depth_) == (256, 8)
    assert len(detector._trees) == 100


def test_c_factor():
    # 2 (H(n - 1)) - 2 (n - 1) / n, with H(i) approximated by ln(i) + Euler's constant.
    assert _c_factor(256) == pytest.approx(2 * (math.log(255) + 0.5772156649) - 2 * 255 / 256)
    assert _c_factor(2) == pytest.approx(2 * 0.5772156649 - 1)


def test_the_outlier_scores_lowest(X_with_outlier):
    detector = ExtendedIsolationForest(random_state=0).fit(X_with_outlier)
    scores = detector.score_samples(X_with_outlier)
    assert np.argmin(scores) == 17
    assert np.all((scores >= -1) & (scores < 0))


@pytest.mark.parametrize("extension_level", [0, 1, 2])
def test_extension_level_sets_the_non_zero_coordinates(X_with_outlier, extension_level):
    detector = ExtendedIsolationForest(
        n_estimators=5, extension_level=extension_level, random_state=0
    ).fit(X_with_outlier)
    for tree in detector._trees:
        internal = tree.children[:, 0] >= 0
        assert np.all(np.count_nonzero(tree.normals[internal], axis=1) == extension_level + 1)


def test_trees_stop_at_max_depth(X_with_outlier):
    detector = ExtendedIsolationForest(n_estimators=3, max_depth=2, random_state=0)
    detector.fit(X_with_outlier)
    assert detector.max_depth_ == 2
    for tree in detector._trees:  # at most 1 + 2 internal nodes above depth 2
        assert np.count_nonzero(tree.children[:, 0] >= 0) <= 3
    # At depth 0 the root is a leaf: every sample gets the average path length of 256 samples.
    stump = ExtendedIsolationForest(n_estimators=3, max_depth=0, random_state=0)
    paths = stump.fit(X_with_outlier)._trees[0].path_lengths(X_with_outlier)
    np.testing.assert_array_equal(paths, np.full(400, _c_factor(256)))


def test_max_samples(X_with_outlier):
    small = X_with_outlier[:100]
    detector = ExtendedIsolationForest(n_estimators=3, random_state=0).fit(small)
    assert (detector.max_samples_, detector.max_depth_) == (100, math.ceil(math.log2(100)))
    with pytest.warns(UserWarning, match="each tree uses all of them"):
        detector = ExtendedIsolationForest(n_estimators=3, max_samples=500).fit(small)
    assert detector.max_samples_ == 100


def test_same_seed_same_scores(X_with_outlier):
    first, again, other = (
        ExtendedIsolationForest(n_estimators=10, random_state=seed).fit(X_with_outlier)
        for seed in (0, 0, 1)
    )
    np.testing.assert_array_equal(
        first.score_samples(X_with_outlier), again.score_samples(X_with_outlier)
    )
    assert not np.array_equal(
        first.score_samples(X_with_outlier), other.score_samples(X_with_outlier)
    )


def test_scores_new_samples_without_refitting(X_with_outlier):
    detector = ExtendedIsolationForest(n_estimators=10, random_state=0).fit(X_with_outlier)
    np.testing.assert_array_equal(
        detector.score_samples(X_with_outlier[:5]), detector.score_samples(X_with_outlier)[:5]
    )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"extension_level": 3}, "at most n_features - 1, got 3 with n_features = 3"),
        ({"extension_level": -1}, "extension_level must be an int >= 0"),
        ({"extension_level": True}, "extension_level must be an int >= 0"),
        ({"n_estimators": 0}, "n_estimators must be an int >= 1"),
        ({"max_samples": 1}, "max_samples must be 'auto' or an int >= 2"),
        ({"max_samples": "all"}, "max_samples must be 'auto' or an int >= 2"),
        ({"max_depth": 2.5}, "max_depth must be None or an int >= 0"),
    ],
)
def test_rejects_invalid_parameters(X_with_outlier, params, message):
    with pytest.raises(ValueError, match=message):
        ExtendedIsolationForest(**params).fit(X_with_outlier)


def test_needs_fitting_and_the_same_features(X_with_outlier):
    detector = ExtendedIsolationForest(n_estimators=3)
    with pytest.raises(NotFittedError):
        detector.score_samples(X_with_outlier)
    clone(detector).fit(X_with_outlier)
    with pytest.raises(ValueError, match="features"):
        detector.fit(X_with_outlier).score_samples(X_with_outlier[:, :2])


@parametrize_with_checks([ExtendedIsolationForest(n_estimators=5, random_state=0)])
def test_sklearn_compatible(estimator, check):
    check(estimator)
