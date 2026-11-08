from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from pypu.models import RankPruning

REFERENCE = Path(__file__).with_name("data") / "rank_pruning_reference.npz"
CASES = ["estimated", "estimated_seed3", "known_label_frequency"]


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


@pytest.mark.parametrize("case", CASES)
def test_reproduces_the_original(reference, case):
    # The original in its PU mode, after np.random.seed(seed); under the same scikit-learn the
    # results agree to the last bit.
    X, s = reference["X_train"], reference["s"]
    prior = s.mean() / 0.4 if case == "known_label_frequency" else None
    model = RankPruning(prior=prior, random_state=int(reference[f"{case}_seed"])).fit(X, s)
    assert 1 - model.label_frequency_ == pytest.approx(float(reference[f"{case}_rh1"]), rel=1e-9)
    assert model.prior_ == pytest.approx(float(reference[f"{case}_py1"]), rel=1e-9)
    assert model.pollution_ == pytest.approx(float(reference[f"{case}_pi0"]), rel=1e-9)
    np.testing.assert_allclose(
        model.predict_proba(reference["X_test"])[:, 1], reference[f"{case}_proba"], atol=1e-6
    )


def test_prunes_the_top_scored_unlabeled_samples(pu_data):
    X, _, s = pu_data
    model = RankPruning(random_state=0).fit(X, s)
    unlabeled = s == 0
    assert not model.pruned_[s == 1].any()
    expected = int(unlabeled.sum() * model.pollution_)
    assert model.pruned_.sum() == expected
    scores = model.oof_proba_
    assert scores[model.pruned_].min() > scores[unlabeled & ~model.pruned_].max()


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Logistic regression that keeps the labels and weights of its last fit."""

    def fit(self, X, y, sample_weight=None):
        self.y_, self.sample_weight_ = y, sample_weight
        self.model_ = LogisticRegression().fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def test_refit_weights_the_labeled_samples(pu_data):
    X, _, s = pu_data
    model = RankPruning(RecordingClassifier(), random_state=0).fit(X, s)
    final = model.estimator_
    np.testing.assert_allclose(final.sample_weight_[final.y_ == 1], 1 / model.label_frequency_)
    np.testing.assert_allclose(final.sample_weight_[final.y_ == 0], 1.0)
    assert len(final.y_) == len(s) - model.pruned_.sum()


def test_prior_gives_the_label_frequency(pu_data):
    X, y, s = pu_data
    model = RankPruning(prior=y.mean(), random_state=0).fit(X, s)
    assert model.label_frequency_ == pytest.approx(s.mean() / y.mean())
    assert model.prior_ == pytest.approx(y.mean())


def test_estimates_the_prior(pu_data):
    X, y, s = pu_data
    model = RankPruning(random_state=0).fit(X, s)
    assert model.prior_ == pytest.approx(y.mean(), abs=0.1)


def test_keeps_at_least_ten_unlabeled_samples():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    s = np.r_[np.ones(31, dtype=int), np.zeros(9, dtype=int)]
    with pytest.warns(UserWarning, match="keeps at least 10 unlabeled samples"):
        model = RankPruning(prior=0.95, random_state=0).fit(X, s)
    assert not model.pruned_.any()


def test_random_state_sets_the_folds(pu_data):
    X, _, s = pu_data
    first = RankPruning(random_state=0).fit(X, s).oof_proba_
    np.testing.assert_array_equal(RankPruning(random_state=0).fit(X, s).oof_proba_, first)
    assert not np.array_equal(RankPruning(random_state=1).fit(X, s).oof_proba_, first)


@pytest.mark.parametrize(
    ("estimator", "match"),
    [(LinearSVC(), "predict_proba"), (KNeighborsClassifier(), "sample_weight")],
)
def test_estimator_requirements(pu_data, estimator, match):
    X, _, s = pu_data
    with pytest.raises(TypeError, match=match):
        RankPruning(estimator).fit(X, s)
