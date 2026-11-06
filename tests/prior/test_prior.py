import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

from pypu.datasets import label_scar, make_case_control
from pypu.prior import BBEPrior, ElkanNotoPrior

ESTIMATORS = [
    BBEPrior,
    ElkanNotoPrior,
    lambda **kw: ElkanNotoPrior(method="e2", **kw),
    lambda **kw: ElkanNotoPrior(method="e3", **kw),
]


class ZeroScores(ClassifierMixin, BaseEstimator):
    """Classifier that gives every sample P(s = 1 | x) = 0."""

    def fit(self, X, s):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


@pytest.fixture(scope="module")
def separated():
    """Positives and negatives in distant blobs; prior 0.3, half the positives labeled."""
    X, y = make_blobs(n_samples=[1400, 600], centers=[[0, 0], [8, 8]], random_state=0)
    s = label_scar(y, label_frequency=0.5, random_state=0)
    return X, y, s


@pytest.fixture(scope="module")
def overlapping():
    X, y = make_classification(
        n_samples=3000,
        n_features=10,
        n_informative=5,
        weights=[0.7],
        class_sep=0.5,
        flip_y=0,
        random_state=0,
    )
    return X, y, label_scar(y, label_frequency=0.5, random_state=0)


@pytest.mark.parametrize("make_estimator", ESTIMATORS)
class TestCommonBehaviour:
    def test_estimates_are_consistent(self, separated, make_estimator):
        X, _, s = separated
        estimator = make_estimator(random_state=0).fit(X, s)
        check_is_fitted(estimator)
        n_labeled, n_unlabeled = s.sum(), (s == 0).sum()
        assert 0 < estimator.pollution_ < 1
        assert s.mean() <= estimator.prior_ < 1
        assert estimator.prior_ * estimator.label_frequency_ == pytest.approx(s.mean())
        assert estimator.pollution_ == pytest.approx(
            (estimator.prior_ * s.size - n_labeled) / n_unlabeled
        )

    def test_same_seed_same_estimate(self, overlapping, make_estimator):
        X, _, s = overlapping
        first = make_estimator(random_state=1).fit(X, s).prior_
        assert make_estimator(random_state=1).fit(X, s).prior_ == first

    def test_accepts_a_cv_splitter_and_dataframes(self, separated, make_estimator):
        X, _, s = separated
        frame = pd.DataFrame(X, columns=["a", "b"])
        estimator = make_estimator(cv=KFold(3, shuffle=True, random_state=0)).fit(frame, s)
        assert list(estimator.feature_names_in_) == ["a", "b"]
        assert estimator.n_features_in_ == 2

    def test_clips_impossible_estimates_with_a_warning(self, separated, make_estimator):
        X, _, s = separated
        n_unlabeled = (s == 0).sum()
        with pytest.warns(UserWarning, match="it is clipped"):
            estimator = make_estimator(estimator=ZeroScores()).fit(X, s)
        assert estimator.pollution_ == pytest.approx((n_unlabeled - 1) / n_unlabeled)
        assert estimator.prior_ < 1

    def test_rejects_estimator_without_predict_proba(self, separated, make_estimator):
        X, _, s = separated
        with pytest.raises(TypeError, match="must implement predict_proba"):
            make_estimator(estimator=LinearSVC()).fit(X, s)

    def test_validates_inputs(self, separated, make_estimator):
        X, _, s = separated
        with pytest.raises(ValueError, match="no labeled positive"):
            make_estimator().fit(X, np.zeros_like(s))
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            make_estimator().fit(X, s[:-1])

    def test_can_be_cloned(self, make_estimator):
        estimator = make_estimator(random_state=3)
        assert clone(estimator).get_params() == estimator.get_params()


class TestAccuracy:
    @pytest.mark.parametrize("method", ["e1", "e2"])
    def test_elkan_noto_without_overlap(self, separated, method):
        X, y, s = separated
        estimator = ElkanNotoPrior(method=method, random_state=0).fit(X, s)
        assert estimator.prior_ == pytest.approx(y.mean(), abs=0.02)
        assert estimator.label_frequency_ == pytest.approx(s[y == 1].mean(), abs=0.04)

    def test_elkan_noto_e3_gives_the_largest_label_frequency(self, overlapping):
        X, _, s = overlapping
        priors = {
            method: ElkanNotoPrior(method=method, random_state=0).fit(X, s).prior_
            for method in ("e1", "e3")
        }
        assert priors["e3"] < priors["e1"]

    def test_bbe_without_overlap(self, separated):
        X, y, s = separated
        assert BBEPrior(random_state=0).fit(X, s).prior_ == pytest.approx(y.mean(), abs=0.02)

    def test_bbe_copes_better_with_overlap(self, overlapping):
        X, y, s = overlapping
        bbe = BBEPrior(random_state=0).fit(X, s).prior_
        elkan_noto = ElkanNotoPrior(random_state=0).fit(X, s).prior_
        assert elkan_noto > y.mean()  # Elkan-Noto overestimates the prior under overlap
        assert abs(bbe - y.mean()) < abs(elkan_noto - y.mean())

    def test_bbe_pollution_is_the_prior_in_case_control_data(self):
        X, y = make_blobs(n_samples=[1400, 600], centers=[[0, 0], [8, 8]], random_state=1)
        X_cc, s_cc, y_cc = make_case_control(X, y, label_frequency=0.4, random_state=0)
        estimator = BBEPrior(random_state=0).fit(X_cc, s_cc)
        assert estimator.pollution_ == pytest.approx(y_cc[s_cc == 0].mean(), abs=0.03)


class TestParameterValidation:
    def test_unknown_elkan_noto_method(self, separated):
        X, _, s = separated
        with pytest.raises(ValueError, match="method must be one of"):
            ElkanNotoPrior(method="e4").fit(X, s)

    @pytest.mark.parametrize("delta", [0, 1, -0.1, "0.1", True])
    def test_bbe_delta(self, separated, delta):
        X, _, s = separated
        with pytest.raises(ValueError, match="delta must be"):
            BBEPrior(delta=delta).fit(X, s)

    @pytest.mark.parametrize("gamma", [-0.1, float("inf"), float("nan"), "0", True])
    def test_bbe_gamma(self, separated, gamma):
        X, _, s = separated
        with pytest.raises(ValueError, match="gamma must be"):
            BBEPrior(gamma=gamma).fit(X, s)
