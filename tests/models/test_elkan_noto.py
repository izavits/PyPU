import numpy as np
import pytest
from scipy.special import logit
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from pypu.datasets import label_scar
from pypu.models import ElkanNoto, NaivePU
from pypu.models.elkan_noto import _unlabeled_weights
from pypu.prior import ElkanNotoPrior
from pypu.prior._base import out_of_fold_scores


@pytest.fixture(scope="module")
def scar_data():
    """Overlapping classes, prior about 0.3, 40% of the positives labeled at random."""
    X, y = make_classification(
        n_samples=3000,
        n_features=6,
        n_informative=4,
        weights=[0.7],
        class_sep=1.5,
        random_state=0,
    )
    s = label_scar(y, label_frequency=0.4, random_state=0)
    return X, y, s


def test_label_frequency_is_the_e1_estimate(scar_data):
    X, _, s = scar_data
    model = ElkanNoto(random_state=0).fit(X, s)
    prior_estimator = ElkanNotoPrior(method="e1", random_state=0).fit(X, s)
    np.testing.assert_allclose(model.label_frequency_, prior_estimator.label_frequency_)


def test_label_frequency_from_the_prior(pu_data):
    X, _, s = pu_data
    model = ElkanNoto(prior=0.3).fit(X, s)
    assert model.label_frequency_ == pytest.approx(s.mean() / 0.3)


def test_adjusted_probability_is_g_over_c(pu_data):
    X, _, s = pu_data
    model = ElkanNoto(prior=0.3).fit(X, s)
    g = LogisticRegression().fit(X, s).predict_proba(X)[:, 1]
    c = model.label_frequency_
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], np.clip(g / c, 0, 1))
    np.testing.assert_allclose(model.decision_function(X), logit(g) - logit(c / 2))


def test_decision_function_keeps_ranking_the_clipped_samples(scar_data):
    X, _, s = scar_data
    model = ElkanNoto(random_state=0).fit(X, s)
    proba, decision = model.predict_proba(X)[:, 1], model.decision_function(X)
    assert (proba == 1).mean() > 0.05  # many samples have g(x) >= c
    g = NaivePU().fit(X, s).predict_proba(X)[:, 1]
    np.testing.assert_array_equal(np.argsort(decision, kind="stable"), np.argsort(g, kind="stable"))
    np.testing.assert_array_equal(decision > 0, proba > 0.5)


def test_prior_below_the_labeled_fraction_gives_the_naive_model(pu_data):
    X, _, s = pu_data
    with pytest.warns(UserWarning, match="lower than the fraction of labeled"):
        model = ElkanNoto(prior=0.01).fit(X, s)
    assert model.label_frequency_ == 1.0
    np.testing.assert_allclose(model.predict_proba(X), NaivePU().fit(X, s).predict_proba(X))


def test_weighted_trains_on_weighted_copies_of_the_unlabeled_samples(pu_data):
    X, _, s = pu_data
    model = ElkanNoto(weighted=True, random_state=0).fit(X, s)

    scores = out_of_fold_scores(LogisticRegression(), X, s, cv=5, random_state=0)
    labeled = s == 1
    c = scores[labeled].mean()
    weight = np.clip((1 - c) / c * scores[~labeled] / (1 - scores[~labeled]), 0, 1)
    X_u = X[~labeled]
    expected = LogisticRegression().fit(
        np.concatenate([X[labeled], X_u, X_u]),
        np.repeat([1, 1, 0], [labeled.sum(), len(X_u), len(X_u)]),
        sample_weight=np.concatenate([np.ones(labeled.sum()), weight, 1 - weight]),
    )
    assert model.label_frequency_ == pytest.approx(c)
    np.testing.assert_allclose(model.predict_proba(X), expected.predict_proba(X), atol=1e-6)


def test_unlabeled_weights():
    scores = np.array([0.0, 0.2, 0.5, 1.0])
    np.testing.assert_allclose(_unlabeled_weights(scores, 0.5), [0.0, 0.25, 1.0, 1.0])
    np.testing.assert_allclose(_unlabeled_weights(scores, 0.25), [0.0, 0.75, 1.0, 1.0])
    np.testing.assert_array_equal(_unlabeled_weights(scores, 1.0), np.zeros(4))


@pytest.mark.parametrize("weighted", [False, True])
def test_calibrated_given_the_prior(scar_data, weighted):
    X, y, s = scar_data
    model = ElkanNoto(weighted=weighted, prior=y.mean(), random_state=0).fit(X, s)
    assert abs(model.predict_proba(X)[:, 1].mean() - y.mean()) < 0.03


@pytest.mark.parametrize("weighted", [False, True])
def test_beats_the_naive_model_at_threshold_one_half(scar_data, weighted):
    X, y, s = scar_data
    model = ElkanNoto(weighted=weighted, random_state=0).fit(X, s)
    naive = NaivePU().fit(X, s)
    assert f1_score(y, model.predict(X)) > f1_score(y, naive.predict(X)) + 0.3


def test_out_of_fold_scores_are_only_computed_when_needed(pu_data):
    X, _, s = pu_data
    ElkanNoto(prior=0.3, cv=10_000).fit(X, s)  # the folds are never made
    with pytest.raises(ValueError, match="n_splits"):
        ElkanNoto(prior=0.3, weighted=True, cv=10_000).fit(X, s)


def test_clips_a_label_frequency_that_implies_a_prior_of_one(pu_data):
    X, _, s = pu_data
    # Scores that cannot tell labeled from unlabeled samples give c close to P(s = 1).
    with pytest.warns(UserWarning, match="estimated label frequency .* is clipped"):
        model = ElkanNoto(DummyClassifier(strategy="prior"), random_state=0).fit(X, s)
    assert model.label_frequency_ == pytest.approx(s.sum() / (s.size - 1))


def test_weighted_needs_sample_weight(pu_data):
    X, _, s = pu_data
    ElkanNoto(KNeighborsClassifier(), random_state=0).fit(X, s)
    with pytest.raises(TypeError, match="sample_weight"):
        ElkanNoto(KNeighborsClassifier(), weighted=True).fit(X, s)


def test_estimator_needs_predict_proba(pu_data):
    X, _, s = pu_data
    with pytest.raises(TypeError, match="predict_proba"):
        ElkanNoto(LinearSVC()).fit(X, s)


def test_rejects_an_invalid_prior(pu_data):
    X, _, s = pu_data
    with pytest.raises(ValueError, match="open interval"):
        ElkanNoto(prior=1.5).fit(X, s)
