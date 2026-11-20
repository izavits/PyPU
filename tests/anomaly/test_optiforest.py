from pathlib import Path

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pypu.anomaly import OptIForest
from pypu.anomaly._optiforest import _GroupNode

REFERENCE = Path(__file__).with_name("data") / "optiforest_reference.npz"

# Cases of data/make_optiforest_reference.py -> (seed, the matching parameters).
SMALL = {"n_estimators": 10, "max_samples": 128, "threshold": 20}
EXACT_CASES = {
    "experiments": (0, {}),
    "l1_pairs": (1, {**SMALL, "lsh_family": "L1OPT", "branch": 2}),
    "angle_random_merges": (2, {**SMALL, "lsh_family": "ALOPT", "branch": 0}),
    "triples_granularity0": (3, {**SMALL, "branch": 4, "granularity": 0}),
}


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture
def X_with_outlier():
    X = np.random.default_rng(0).normal(size=(300, 3))
    X[17] = 6.0
    return X


@pytest.mark.parametrize("case", list(EXACT_CASES))
def test_reproduces_the_original_from_the_same_seed(reference, case):
    seed, params = EXACT_CASES[case]
    detector = OptIForest(random_state=seed, **params).fit(reference["X"])
    # Equal on the macOS machine that saved the reference. Elsewhere NumPy's log and power can
    # differ in the last bits; one sample sent the other way in one tree would change its
    # score by far more than this tolerance.
    np.testing.assert_allclose(
        detector.score_samples(reference["X"]), reference[f"{case}_scores"], rtol=1e-12, atol=0
    )


def test_defaults_are_the_experiments_settings():
    # main_2step_ad.py: OptIForest('L2OPT', 100, 403, 20, 1)
    params = OptIForest().get_params()
    assert params == {
        "n_estimators": 100,
        "max_samples": 512,
        "lsh_family": "L2OPT",
        "threshold": 403,
        "branch": 20,
        "granularity": 1,
        "random_state": None,
    }


def test_without_a_seed_it_draws_from_numpys_global_generator(X_with_outlier):
    np.random.seed(5)  # noqa: NPY002 - what the original relies on
    unseeded = OptIForest(n_estimators=5).fit(X_with_outlier).score_samples(X_with_outlier)
    seeded = OptIForest(n_estimators=5, random_state=5).fit(X_with_outlier)
    np.testing.assert_array_equal(unseeded, seeded.score_samples(X_with_outlier))


def test_the_outlier_scores_lowest(X_with_outlier):
    detector = OptIForest(n_estimators=20, random_state=0).fit(X_with_outlier)
    scores = detector.score_samples(X_with_outlier)
    assert np.argmin(scores) == 17
    assert np.all((scores >= -1) & (scores < 0))


def test_max_samples_is_lowered_to_the_number_of_samples(X_with_outlier):
    detector = OptIForest(n_estimators=3, random_state=0).fit(X_with_outlier)
    assert detector.max_samples_ == 300
    # 300 samples are within the threshold: the trees are not regrouped.
    assert not any(isinstance(tree.root, _GroupNode) for tree in detector._trees)


@pytest.mark.parametrize("branch", [2, 3, 5])
def test_the_new_root_has_at_most_branch_children(X_with_outlier, branch):
    detector = OptIForest(n_estimators=3, threshold=60, branch=branch, random_state=0)
    for tree in detector.fit(X_with_outlier)._trees:
        assert isinstance(tree.root, _GroupNode)
        assert 2 <= len(tree.root.children) <= branch


def test_same_seed_same_scores(X_with_outlier):
    first, again, other = (
        OptIForest(n_estimators=5, random_state=seed)
        .fit(X_with_outlier)
        .score_samples(X_with_outlier)
        for seed in (0, 0, 1)
    )
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_scores_new_samples_without_refitting(X_with_outlier):
    detector = OptIForest(n_estimators=5, threshold=60, random_state=0).fit(X_with_outlier)
    # SIMD code may round the tail of an array differently: equal up to rounding.
    np.testing.assert_allclose(
        detector.score_samples(X_with_outlier[:7]),
        detector.score_samples(X_with_outlier)[:7],
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"lsh_family": "L3OPT"}, "lsh_family must be one of 'L2OPT', 'L1OPT', 'ALOPT'"),
        ({"branch": 1}, "branch must be 0 or an int >= 2"),
        ({"branch": -1}, "branch must be an int >= 0"),
        ({"threshold": 0}, "threshold must be an int >= 1"),
        ({"max_samples": 1}, "max_samples must be an int >= 2"),
        ({"n_estimators": True}, "n_estimators must be an int >= 1"),
        ({"granularity": -1}, "granularity must be a number >= 0"),
        ({"granularity": True}, "granularity must be a number >= 0"),
    ],
)
def test_rejects_invalid_parameters(X_with_outlier, params, message):
    with pytest.raises(ValueError, match=message):
        OptIForest(**params).fit(X_with_outlier)


@parametrize_with_checks([OptIForest(n_estimators=5, random_state=0)])
def test_sklearn_compatible(estimator, check):
    check(estimator)
