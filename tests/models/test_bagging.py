from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.random import sample_without_replacement

from pypu.models import PUBagging

REFERENCE = Path(__file__).with_name("data") / "bagging_reference.npz"

# Reference cases of data/make_bagging_reference.py -> (matching PUBagging parameters,
# tolerance on the test scores). Both implementations build the same bags, but scikit-learn
# changed since the reference was made (1.0): one of the ten fully grown trees of
# "tree_defaults" comes out differently from the same weighted bag, and LogisticRegression
# itself moved by about 1e-3.
REFERENCE_CASES = {
    "tree_defaults": ({"n_estimators": 10, "max_samples": 1.0, "random_state": 0}, 0.1),
    "svm_experiments": (
        {"estimator": SVC(kernel="linear"), "n_estimators": 20, "random_state": 3},
        0.0,
    ),
    "tree_subsampled": (
        {
            "estimator": DecisionTreeClassifier(max_depth=3),
            "n_estimators": 15,
            "max_samples": 0.3,
            "max_features": 0.5,
            "bootstrap": False,
            "random_state": 7,
        },
        0.0,
    ),
    "knn_no_weights": (
        {"estimator": KNeighborsClassifier(), "n_estimators": 10, "random_state": 1},
        0.0,
    ),
    "logreg_bootstrap_features": (
        {
            "estimator": LogisticRegression(),
            "n_estimators": 10,
            "max_samples": 1.0,
            "max_features": 4,
            "bootstrap_features": True,
            "random_state": 5,
        },
        1e-3,
    ),
}


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


def fit_reference(reference, case, **extra):
    params, _ = REFERENCE_CASES[case]
    return PUBagging(**params, **extra).fit(reference["X_train"], reference["s"])


@pytest.mark.parametrize("case", list(REFERENCE_CASES))
def test_builds_the_original_bags(reference, case):
    model = fit_reference(reference, case)
    np.testing.assert_array_equal(model.estimators_features_, reference[f"{case}_features"])
    random_states = [getattr(member, "random_state", -1) for member in model.estimators_]
    np.testing.assert_array_equal(random_states, reference[f"{case}_random_states"])


@pytest.mark.parametrize("case", list(REFERENCE_CASES))
def test_reproduces_the_original_scores(reference, case):
    model = fit_reference(reference, case)
    _, tolerance = REFERENCE_CASES[case]
    np.testing.assert_allclose(
        model.predict_proba(reference["X_test"])[:, 1],
        reference[f"{case}_scores"],
        rtol=0,
        atol=tolerance + 1e-12,
    )


def test_out_of_bag_matches_the_original_for_classifiers_without_random_state(reference):
    model = fit_reference(reference, "knn_no_weights", oob_proba=True)
    np.testing.assert_allclose(
        model.oob_proba_, reference["knn_no_weights_oob"], rtol=0, atol=1e-12, equal_nan=True
    )


def test_original_out_of_bag_used_other_bags(reference):
    # The original redrew each bag from a fresh RandomState(seed) without first setting the
    # classifier's random_state, which shifts the draws; scoring PyPU's classifiers on those
    # bags gives back its out-of-bag estimates.
    model = fit_reference(reference, "svm_experiments", oob_proba=True)
    X, s = reference["X_train"], reference["s"]
    labeled, unlabeled = np.flatnonzero(s == 1), np.flatnonzero(s == 0)
    seeds = np.random.RandomState(3).randint(np.iinfo(np.int32).max, size=20)
    votes = np.zeros((s.size, 2))
    for seed, member, features in zip(
        seeds, model.estimators_, model.estimators_features_, strict=True
    ):
        random_state = np.random.RandomState(seed)
        sample_without_replacement(X.shape[1], X.shape[1], random_state=random_state)
        drawn = unlabeled[random_state.randint(0, unlabeled.size, labeled.size)]
        out_of_bag = np.ones(s.size, dtype=bool)
        out_of_bag[np.concatenate([drawn, labeled])] = False
        predicted = member.predict(X[out_of_bag][:, features])
        votes[np.flatnonzero(out_of_bag), predicted] += 1
    with np.errstate(invalid="ignore"):
        original = votes / votes.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(original, reference["svm_experiments_oob"], equal_nan=True)
    assert not np.allclose(model.oob_proba_, original, equal_nan=True)


def test_default_bags_are_balanced(pu_data):
    X, _, s = pu_data
    model = PUBagging(KNeighborsClassifier(), n_estimators=5, random_state=0).fit(X, s)
    # KNeighborsClassifier.fit takes no sample_weight, so it is trained on the bag itself.
    assert {member.n_samples_fit_ for member in model.estimators_} == {2 * s.sum()}


def test_default_max_samples_is_capped_by_the_unlabeled_samples(pu_data):
    X, _, _ = pu_data
    s = np.zeros(X.shape[0], dtype=int)
    s[:300] = 1  # more labeled than unlabeled samples
    model = PUBagging(KNeighborsClassifier(), n_estimators=3, random_state=0).fit(X, s)
    assert {member.n_samples_fit_ for member in model.estimators_} == {X.shape[0]}


def test_without_bootstrap_the_unlabeled_samples_are_drawn_once(pu_data):
    X, _, s = pu_data
    model = PUBagging(
        KNeighborsClassifier(), n_estimators=3, max_samples=1.0, bootstrap=False, random_state=0
    ).fit(X, s)
    for member in model.estimators_:
        assert len(np.unique(member._fit_X, axis=0)) == X.shape[0]


def test_n_jobs_does_not_change_the_result(pu_data):
    X, _, s = pu_data
    serial = PUBagging(n_estimators=8, random_state=0).fit(X, s)
    parallel = PUBagging(n_estimators=8, random_state=0, n_jobs=2).fit(X, s)
    np.testing.assert_array_equal(serial.predict_proba(X), parallel.predict_proba(X))


def test_classifiers_without_predict_proba_vote(pu_data):
    X, y, s = pu_data
    model = PUBagging(LinearSVC(), n_estimators=7, random_state=0).fit(X, s)
    votes = model.predict_proba(X)[:, 1] * 7
    np.testing.assert_allclose(votes, np.round(votes), atol=1e-12)
    assert roc_auc_score(y, votes) > 0.7


def test_out_of_bag_proba(pu_data):
    X, y, s = pu_data
    model = PUBagging(n_estimators=30, oob_proba=True, random_state=0).fit(X, s)
    labeled = s == 1
    assert model.oob_proba_.shape == (X.shape[0], 2)
    assert np.isnan(model.oob_proba_[labeled]).all()  # the labeled positives are in every bag
    unlabeled_proba = model.oob_proba_[~labeled]
    assert not np.isnan(unlabeled_proba).any()
    np.testing.assert_allclose(unlabeled_proba.sum(axis=1), 1.0)
    assert roc_auc_score(y[~labeled], unlabeled_proba[:, 1]) > 0.8


def test_out_of_bag_proba_is_opt_in(pu_data):
    X, _, s = pu_data
    assert not hasattr(PUBagging(n_estimators=3).fit(X, s), "oob_proba_")


def test_classifier_random_states_come_from_random_state(pu_data):
    X, _, s = pu_data
    first = PUBagging(n_estimators=5, random_state=0).fit(X, s)
    second = PUBagging(n_estimators=5, random_state=0).fit(X, s)
    states = [member.random_state for member in first.estimators_]
    assert len(set(states)) == 5
    assert states == [member.random_state for member in second.estimators_]


@pytest.mark.parametrize(
    ("max_features", "bootstrap_features", "n_drawn"),
    [(0.5, False, 4), (3, False, 3), (8, True, 8)],
)
def test_feature_draws(pu_data, max_features, bootstrap_features, n_drawn):
    X, _, s = pu_data
    model = PUBagging(
        n_estimators=5,
        max_features=max_features,
        bootstrap_features=bootstrap_features,
        random_state=0,
    ).fit(X, s)
    for features in model.estimators_features_:
        assert features.shape == (n_drawn,)
        if not bootstrap_features:
            assert len(np.unique(features)) == n_drawn
    if bootstrap_features:  # with replacement, some draws repeat a feature
        assert any(len(np.unique(f)) < n_drawn for f in model.estimators_features_)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"n_estimators": 0}, "n_estimators must be a positive int"),
        ({"n_estimators": 2.5}, "n_estimators must be a positive int"),
        ({"n_estimators": True}, "n_estimators must be a positive int"),
        ({"max_samples": 0}, "max_samples=0 must be between 1 and the number of unlabeled"),
        ({"max_samples": 10_000}, "max_samples=10000 must be between 1"),
        ({"max_samples": 1.5}, r"max_samples must be in \(0, 1\]"),
        ({"max_samples": 1e-4}, "selects no unlabeled samples"),
        ({"max_samples": "half"}, "max_samples must be an int, a float or None"),
        ({"max_features": 0}, "max_features=0 must be between 1 and the number of features"),
        ({"max_features": 9}, r"the number of features \(8\)"),
        ({"max_features": None}, "max_features must be an int or a float"),
    ],
)
def test_invalid_parameters_raise(pu_data, params, match):
    X, _, s = pu_data
    with pytest.raises(ValueError, match=match):
        PUBagging(**{"n_estimators": 2, **params}).fit(X, s)
