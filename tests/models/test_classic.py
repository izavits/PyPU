"""Tests of the classic methods: RebalancedPU, PropensityWeightedPU, BiasedSVM, SpyPU, PEBL.

The reference procedures below follow the notebooks of github.com/hkiyomaru/pu-learning
(MIT License), on data like theirs: two Gaussian classes in two dimensions.
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from pypu.datasets import label_scar
from pypu.metrics import lee_liu_score
from pypu.models import PEBL, BiasedSVM, PropensityWeightedPU, RebalancedPU, SpyPU


@pytest.fixture(scope="module")
def gaussians():
    """The notebooks' setting: positives around (3, 3), negatives around (0, 0), SCAR labels."""
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.6).astype(int)
    X = rng.normal(size=(3000, 2)) + 3.0 * y[:, None]
    s = label_scar(y, label_frequency=0.2, random_state=0)
    return X, y, s


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Logistic regression that keeps the data and weights it was fitted on."""

    def fit(self, X, y, sample_weight=None):
        self.X_, self.y_, self.sample_weight_ = X, y, sample_weight
        self.model_ = LogisticRegression().fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


# --- RebalancedPU -----------------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.5, 0.3])
def test_rebalancing_weights_are_elkans(pu_data, tau):
    X, y, s = pu_data
    model = RebalancedPU(RecordingClassifier(), prior=y.mean(), threshold=tau).fit(X, s)
    c = s.mean() / y.mean()
    expected = np.where(s == 1, 1 - c * tau, (1 - tau) * c)
    np.testing.assert_allclose(model.estimator_.sample_weight_, expected)
    assert model.label_frequency_ == pytest.approx(c)
    assert model.threshold_ == tau


def test_rebalancing_classifies_like_a_classifier_of_the_true_labels(gaussians):
    # The weights move the threshold exactly; a logistic regression cannot represent
    # P(s = 1 | x) = c P(y = 1 | x) exactly, so the decisions agree closely, not perfectly.
    X, y, s = gaussians
    traditional = LogisticRegression().fit(X, y).predict(X)
    rebalanced = RebalancedPU(prior=y.mean()).fit(X, s).predict(X)
    unweighted = LogisticRegression().fit(X, s).predict(X)
    assert np.mean(rebalanced == traditional) > 0.9
    assert np.mean(rebalanced == traditional) > np.mean(unweighted == traditional) + 0.3


def test_rebalancing_needs_the_prior_and_a_valid_threshold(pu_data):
    X, _, s = pu_data
    with pytest.raises(ValueError, match="needs the class prior"):
        RebalancedPU().fit(X, s)
    for threshold in (0.0, 1.0, "0.5"):
        with pytest.raises(ValueError, match="threshold must be a number in"):
            RebalancedPU(prior=0.3, threshold=threshold).fit(X, s)


def test_rebalancing_needs_sample_weight_and_predict_proba(pu_data):
    X, _, s = pu_data
    with pytest.raises(TypeError, match="sample_weight"):
        RebalancedPU(KNeighborsClassifier(), prior=0.3).fit(X, s)
    with pytest.raises(TypeError, match="predict_proba"):
        RebalancedPU(LinearSVC(), prior=0.3).fit(X, s)


# --- PropensityWeightedPU ---------------------------------------------------------------


def test_propensity_weighting_matches_the_reference_procedure(pu_data):
    # The notebook's weighted data: labeled samples as positives (1 / e) and negatives
    # (1 - 1 / e), unlabeled samples as negatives (1).
    X, _, s = pu_data
    e = np.linspace(0.3, 1.0, int(s.sum()))
    L, U = X[s == 1], X[s == 0]
    weight = np.concatenate([1 / e, 1 - 1 / e, np.ones(len(U))])
    keep = weight != 0  # PyPU drops the zero-weight copy of e = 1
    X_ref = np.concatenate([L, L, U])[keep]
    y_ref = np.concatenate([np.ones(len(L)), np.zeros(len(L) + len(U))])[keep]
    reference = LogisticRegression(solver="liblinear").fit(X_ref, y_ref, sample_weight=weight[keep])
    model = PropensityWeightedPU(propensity=lambda X_labeled: e).fit(X, s)
    np.testing.assert_allclose(
        model.predict_proba(X)[:, 1], reference.predict_proba(X)[:, 1], rtol=1e-10, atol=1e-12
    )


def test_constant_propensity_equals_a_function(pu_data):
    X, _, s = pu_data
    constant = PropensityWeightedPU(propensity=0.5).fit(X, s).predict_proba(X)
    function = PropensityWeightedPU(propensity=lambda X: np.full(len(X), 0.5)).fit(X, s)
    np.testing.assert_allclose(constant, function.predict_proba(X))


def test_propensity_weighting_with_sarem(pu_data):
    from pypu.models import SAREM

    X, y, s = pu_data
    sarem = SAREM().fit(X, s)
    model = PropensityWeightedPU(propensity=sarem.predict_propensity).fit(X, s)
    assert model.predict(X).shape == y.shape


@pytest.mark.parametrize(
    ("propensity", "error", "match"),
    [
        (None, ValueError, "needs the propensity"),
        (0.0, ValueError, r"must be in \(0, 1\]"),
        (1.5, ValueError, r"must be in \(0, 1\]"),
        (lambda X: np.ones(3), ValueError, "returned 3 values"),
        ("0.5", TypeError, "number or a callable"),
    ],
)
def test_propensity_validation(pu_data, propensity, error, match):
    X, _, s = pu_data
    with pytest.raises(error, match=match):
        PropensityWeightedPU(propensity=propensity).fit(X, s)


# --- BiasedSVM --------------------------------------------------------------------------


def test_biased_svm_chooses_the_weight_with_the_best_criterion(gaussians):
    # The notebook: an SVM per weight, scored by Lee and Liu's criterion on validation data.
    X, _, s = gaussians
    train, valid = np.arange(2000), np.arange(2000, 3000)
    weights = [0.7, 0.8, 0.9]
    expected = [
        lee_liu_score(
            s[valid], SVC(class_weight={0: 1 - w, 1: w}).fit(X[train], s[train]).predict(X[valid])
        )
        for w in weights
    ]
    model = BiasedSVM(positive_weights=weights, cv=[(train, valid)]).fit(X, s)
    np.testing.assert_allclose(model.cv_scores_, expected)
    assert model.positive_weight_ == weights[int(np.argmax(expected))]
    assert model.estimator_.class_weight == {
        0: pytest.approx(1 - model.positive_weight_),
        1: model.positive_weight_,
    }


def test_biased_svm_predict_is_the_svms_classification(gaussians):
    X, _, s = gaussians
    model = BiasedSVM(positive_weights=[0.8], cv=2, random_state=0).fit(X, s)
    np.testing.assert_array_equal(model.predict(X), model.estimator_.predict(X))
    np.testing.assert_allclose(model.decision_function(X), model.estimator_.decision_function(X))


def test_biased_svm_accepts_linear_svc(pu_data):
    X, _, s = pu_data
    model = BiasedSVM(LinearSVC(), positive_weights=[0.8, 0.9], cv=3, random_state=0).fit(X, s)
    assert model.predict_proba(X).shape == (len(X), 2)


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"positive_weights": []}, ValueError, "non-empty sequence"),
        ({"positive_weights": [0.5, 1.0]}, ValueError, r"numbers in \(0, 1\)"),
        ({"estimator": GaussianNB()}, TypeError, "class_weight parameter"),
    ],
)
def test_biased_svm_validation(pu_data, params, error, match):
    X, _, s = pu_data
    with pytest.raises(error, match=match):
        BiasedSVM(**params).fit(X, s)


# --- SpyPU ------------------------------------------------------------------------------


def test_spy_matches_the_reference_procedure(gaussians):
    # The notebook: 10% of the labeled samples as spies (np.random.seed(0)), a logistic
    # regression with the spies unlabeled, reliable negatives below the lowest spy, then a
    # Gaussian naive Bayes on the labeled samples against the reliable negatives.
    X, _, s = gaussians
    labeled = np.flatnonzero(s == 1)
    spies = np.random.RandomState(0).choice(labeled, round(len(labeled) * 0.1), replace=False)
    s_spied = s.copy()
    s_spied[spies] = 0
    proba = LogisticRegression().fit(X, s_spied).predict_proba(X)[:, 1]
    negatives = np.flatnonzero((proba < proba[spies].min()) & (s == 0))
    train = np.concatenate([negatives, labeled])
    reference = GaussianNB().fit(X[train], s[train])

    model = SpyPU(random_state=0).fit(X, s)
    np.testing.assert_array_equal(model.spies_, np.sort(spies))
    assert model.spy_threshold_ == proba[spies].min()
    np.testing.assert_array_equal(np.flatnonzero(model.reliable_negatives_), negatives)
    np.testing.assert_allclose(
        model.predict_proba(X)[:, 1], reference.predict_proba(X)[:, 1], rtol=1e-9, atol=1e-12
    )


def test_spy_quantile_tolerates_outlying_spies(gaussians):
    X, _, s = gaussians
    strict = SpyPU(random_state=0).fit(X, s)
    tolerant = SpyPU(spy_quantile=0.15, random_state=0).fit(X, s)
    assert tolerant.spy_threshold_ >= strict.spy_threshold_
    assert tolerant.reliable_negatives_.sum() >= strict.reliable_negatives_.sum()


def test_spy_em_refines_the_classifier(gaussians):
    X, _, s = gaussians
    plain = SpyPU(random_state=0).fit(X, s)
    refined = SpyPU(n_em_iter=3, random_state=0).fit(X, s)
    assert not np.allclose(plain.predict_proba(X), refined.predict_proba(X))
    np.testing.assert_array_equal(plain.reliable_negatives_, refined.reliable_negatives_)


def test_spy_without_reliable_negatives_warns(pu_data):
    X, _, s = pu_data
    with pytest.warns(UserWarning, match="No unlabeled sample scores below the spies"):
        model = SpyPU(spy_ratio=0.01, spy_quantile=0.0, random_state=0).fit(np.zeros_like(X), s)
    assert model.reliable_negatives_.sum() == 1


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"spy_ratio": 0.0}, ValueError, "spy_ratio must be"),
        ({"spy_quantile": 1.0}, ValueError, "spy_quantile must be"),
        ({"n_em_iter": -1}, ValueError, "n_em_iter must be"),
        ({"estimator": LinearSVC()}, TypeError, "estimator must implement predict_proba"),
        ({"n_em_iter": 1, "final_estimator": KNeighborsClassifier()}, TypeError, "sample_weight"),
    ],
)
def test_spy_validation(pu_data, params, error, match):
    X, _, s = pu_data
    with pytest.raises(error, match=match):
        SpyPU(**params).fit(X, s)


# --- PEBL -------------------------------------------------------------------------------


def test_pebl_spy_negatives_are_those_of_spypu(gaussians):
    X, _, s = gaussians
    pebl = PEBL(random_state=0).fit(X, s)
    spy = SpyPU(random_state=0).fit(X, s)
    np.testing.assert_array_equal(pebl.spies_, spy.spies_)
    np.testing.assert_array_equal(pebl.reliable_negatives_, spy.reliable_negatives_)


def test_pebl_matches_the_reference_iterative_svm(gaussians):
    # The notebook's iterative SVM and selection, from the spy reliable negatives.
    X, _, s = gaussians
    reliable = SpyPU(random_state=0).fit(X, s).reliable_negatives_
    labels = np.where(s == 1, 1, np.where(reliable, 0, -1))
    models = []
    while True:
        model = SVC(kernel="linear").fit(X[labels != -1], labels[labels != -1])
        models.append(model)
        joining = (labels == -1) & (model.predict(X) == 0)
        if not joining.any():
            break
        labels[joining] = 0
    criteria = [lee_liu_score(s, m.predict(X)) for m in models]

    pebl = PEBL(random_state=0).fit(X, s)
    assert pebl.n_iter_ == len(models)
    assert pebl.selected_iteration_ == int(np.argmax(criteria))
    np.testing.assert_array_equal(pebl.predict(X), models[pebl.selected_iteration_].predict(X))


@pytest.fixture(scope="module")
def words():
    """Binary features: words 0-2 mostly in positive documents, 3-5 mostly in negative ones."""
    rng = np.random.default_rng(1)
    y = (rng.random(1500) < 0.4).astype(int)
    p_word = np.where(
        y[:, None] == 1, [0.6, 0.5, 0.4, 0.02, 0.02, 0.02], [0.03, 0.02, 0.02, 0.6, 0.5, 0.4]
    )
    X = (rng.random((1500, 6)) < p_word).astype(float)
    s = label_scar(y, label_frequency=0.3, random_state=1)
    return X, y, s


def test_one_dnf_on_binary_features(words):
    X, y, s = words
    present = X != 0
    positive = present[s == 1].mean(axis=0) > present[s == 0].mean(axis=0)
    reliable = (s == 0) & ~present[:, positive].any(axis=1)
    model = PEBL(negatives="1dnf").fit(X, s)
    np.testing.assert_array_equal(model.positive_features_, positive)
    np.testing.assert_array_equal(model.reliable_negatives_, reliable)
    assert positive[:3].all() and not positive[3:].any()
    assert np.mean(y[model.reliable_negatives_] == 0) > 0.9


def test_one_dnf_on_continuous_features_warns(gaussians):
    X, _, s = gaussians
    with pytest.warns(UserWarning, match="1-DNF found no positive feature"):
        model = PEBL(negatives="1dnf").fit(X + 10, s)  # no zero value: every feature present
    np.testing.assert_array_equal(model.reliable_negatives_, s == 0)


def test_one_dnf_without_reliable_negatives_warns():
    # Both words are in every labeled document and in half the unlabeled ones, which each
    # have one of them.
    X = np.array([[1, 1], [1, 1], [1, 1], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    s = np.array([1, 1, 1, 0, 0, 0, 0])
    with pytest.warns(UserWarning, match="Every unlabeled sample has a positive feature"):
        model = PEBL(negatives="1dnf").fit(X, s)
    assert model.reliable_negatives_.sum() == 1


def test_pebl_selection_last(gaussians):
    X, _, s = gaussians
    model = PEBL(selection="last", random_state=0).fit(X, s)
    assert model.selected_iteration_ == model.n_iter_ - 1
    assert model.estimator_ is model.estimators_[-1]


def test_pebl_max_iter(gaussians):
    X, _, s = gaussians
    assert PEBL(max_iter=1, random_state=0).fit(X, s).n_iter_ == 1


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"negatives": "rocchio"}, ValueError, "negatives must be"),
        ({"selection": "best"}, ValueError, "selection must be"),
        ({"max_iter": 0}, ValueError, "max_iter must be"),
        ({"spy_ratio": 1.0}, ValueError, "spy_ratio must be"),
        ({"spy_estimator": LinearSVC()}, TypeError, "spy_estimator must implement"),
    ],
)
def test_pebl_validation(pu_data, params, error, match):
    X, _, s = pu_data
    with pytest.raises(error, match=match):
        PEBL(**params).fit(X, s)
