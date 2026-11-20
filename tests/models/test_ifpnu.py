import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from pypu.anomaly import ExtendedIsolationForest, OptIForest
from pypu.models import IFPNU
from pypu.models.ifpnu import _anomaly_scores, _FocalPNUObjective, _pnu_labels

REFERENCE = Path(__file__).with_name("data") / "ifpnu_reference.npz"

# Reference cases of data/make_ifpnu_reference.py -> the IFPNU parameters that match them.
REFERENCE_CASES = {
    "published_scar": {},
    "published_sar": {},
    "eta08_gamma1": {"eta": 0.8, "gamma": 1.0, "n_estimators": 30, "learning_rate": 0.05},
}


class FixedScores(BaseEstimator):
    """A stand-in PyOD detector whose training scores are given."""

    def __init__(self, scores=None):
        self.scores = scores

    def fit(self, X, y=None):
        self.decision_scores_ = self.scores
        return self


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


@pytest.mark.parametrize("case", list(REFERENCE_CASES))
def test_reproduces_the_original_implementation(reference, case):
    # The anomaly scores of the reference run, so that this does not depend on the detector.
    model = IFPNU(
        prior=float(reference["prior"]),
        detector=FixedScores(reference["anomaly_scores"]),
        **REFERENCE_CASES[case],
    ).fit(reference["X_train"], reference[f"{case}_s"])
    np.testing.assert_array_equal(model.pnu_labels_, reference[f"{case}_pnu"])
    assert model.booster_.num_trees() == reference[f"{case}_n_trees"]
    np.testing.assert_allclose(
        model.predict_proba(reference["X_test"])[:, 1], reference[f"{case}_scores"], atol=1e-9
    )


def test_default_detector_matches_the_seeded_original(reference):
    # PyOD's IForest wraps scikit-learn's IsolationForest, so both rank samples alike.
    model = IFPNU(prior=float(reference["prior"]), random_state=0)
    model.fit(reference["X_train"], reference["published_scar_s"])
    assert isinstance(model.detector_, IsolationForest)
    assert model.detector_.random_state == 0
    np.testing.assert_array_equal(model.pnu_labels_, reference["published_scar_pnu"])
    np.testing.assert_allclose(
        model.predict_proba(reference["X_test"])[:, 1],
        reference["published_scar_scores"],
        atol=1e-9,
    )


def test_defaults_are_the_published_settings(pu_data):
    X, y, s = pu_data
    model = IFPNU(prior=y.mean(), random_state=0).fit(X, s)
    assert model.booster_.num_trees() == 100
    assert model.booster_.params["learning_rate"] == 0.1
    assert (model.gamma, model.eta) == (3.0, 0.5)


class TestPNULabels:
    def test_most_anomalous_unlabeled_samples_stay_unlabeled(self):
        s = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        scores = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        # prior 0.4 flags the 4 most anomalous samples, two of them labeled.
        labels = _pnu_labels(s, scores, 0.4)
        np.testing.assert_array_equal(labels, [1, -1, -1, 1, 0, 0, 0, 0, 0, 0])

    def test_ties_flag_the_later_samples_first(self):
        s = np.zeros(10, dtype=int)
        s[0] = 1
        labels = _pnu_labels(s, np.zeros(10), 0.3)
        np.testing.assert_array_equal(labels, [1, 0, 0, 0, 0, 0, 0, -1, -1, -1])

    def test_attribute(self, pu_data):
        X, y, s = pu_data
        model = IFPNU(prior=y.mean(), n_estimators=5, random_state=0).fit(X, s)
        labels = model.pnu_labels_
        np.testing.assert_array_equal(labels == 1, s == 1)
        assert np.sum(labels == -1) <= int(len(s) * y.mean())
        assert np.sum(labels == 0) >= np.sum(s == 0) - int(len(s) * y.mean())


class TestAnomalyScores:
    @pytest.fixture
    def X_with_outlier(self):
        X = np.random.default_rng(0).normal(size=(200, 3))
        X[17] = 10.0
        return X

    @pytest.mark.parametrize(
        "detector",
        [IsolationForest(random_state=0), LocalOutlierFactor(), OneClassSVM(gamma=0.1)],
        ids=["score_samples", "negative_outlier_factor_", "one_class_svm"],
    )
    def test_higher_means_more_anomalous(self, X_with_outlier, detector):
        scores = _anomaly_scores(detector.fit(X_with_outlier), X_with_outlier)
        assert scores.shape == (200,)
        assert np.argmax(scores) == 17

    def test_pyod_training_scores_are_used_as_they_are(self, X_with_outlier):
        given = np.arange(200.0)
        detector = FixedScores(given).fit(X_with_outlier)
        np.testing.assert_array_equal(_anomaly_scores(detector, X_with_outlier), given)

    def test_rejects_estimators_that_are_not_detectors(self, pu_data):
        class NotADetector(BaseEstimator):
            def fit(self, X, y=None):
                return self

        X, y, s = pu_data
        with pytest.raises(TypeError, match="PyOD detector or a scikit-learn outlier detector"):
            IFPNU(prior=y.mean(), detector=NotADetector()).fit(X, s)

    @pytest.mark.parametrize(
        "scores", [np.zeros(3), np.full(400, np.nan)], ids=["wrong_length", "nan"]
    )
    def test_rejects_bad_scores(self, pu_data, scores):
        X, y, s = pu_data
        with pytest.raises(ValueError, match="one finite anomaly score per training sample"):
            IFPNU(prior=y.mean(), detector=FixedScores(scores)).fit(X, s)


class TestDetectorNames:
    def test_default_is_a_seeded_isolation_forest(self, pu_data):
        X, y, s = pu_data
        default = IFPNU(prior=y.mean(), n_estimators=5, random_state=0).fit(X, s)
        forest = IsolationForest(random_state=0)
        given = IFPNU(prior=y.mean(), detector=forest, n_estimators=5).fit(X, s)
        assert default.detector == "IF"
        np.testing.assert_array_equal(default.pnu_labels_, given.pnu_labels_)

    @pytest.mark.parametrize("name", ["if", "If", None])
    def test_names_ignore_case_and_none_is_if(self, pu_data, name):
        X, y, s = pu_data
        model = IFPNU(prior=y.mean(), detector=name, n_estimators=5, random_state=0).fit(X, s)
        default = IFPNU(prior=y.mean(), n_estimators=5, random_state=0).fit(X, s)
        assert isinstance(model.detector_, IsolationForest)
        np.testing.assert_array_equal(model.pnu_labels_, default.pnu_labels_)

    def test_eif_is_the_experiments_eif_seeded(self, pu_data):
        X, y, s = pu_data
        model = IFPNU(prior=y.mean(), detector="eif", n_estimators=5, random_state=0).fit(X, s)
        detector = model.detector_
        assert isinstance(detector, ExtendedIsolationForest)
        assert (detector.n_estimators, detector.max_samples_, detector.extension_level) == (
            100,
            256,
            1,
        )
        given = ExtendedIsolationForest(random_state=0).fit(X)
        np.testing.assert_array_equal(
            model.pnu_labels_, _pnu_labels(s, -given.score_samples(X), y.mean())
        )

    def test_oif_is_the_experiments_optiforest_seeded(self, pu_data):
        X, y, s = pu_data
        model = IFPNU(prior=y.mean(), detector="Oif", n_estimators=5, random_state=0).fit(X, s)
        assert isinstance(model.detector_, OptIForest)
        assert model.detector_.get_params() == OptIForest(random_state=0).get_params()
        given = OptIForest(random_state=0).fit(X)
        np.testing.assert_array_equal(
            model.pnu_labels_, _pnu_labels(s, -given.score_samples(X), y.mean())
        )

    def test_lof_matches_the_experiments_pyod_lof(self, pu_data):
        lof = pytest.importorskip("pyod.models.lof")
        X, y, s = pu_data
        model = IFPNU(prior=y.mean(), detector="lof", n_estimators=5).fit(X, s)
        assert isinstance(model.detector_, LocalOutlierFactor)
        # As run: LOF(contamination=pu_ratio, n_jobs=-1); contamination leaves the scores.
        original = lof.LOF(contamination=0.25, n_jobs=-1).fit(X)
        np.testing.assert_array_equal(
            model.pnu_labels_, _pnu_labels(s, original.decision_scores_, y.mean())
        )

    @pytest.mark.filterwarnings("ignore:'pin_memory' argument is set as true")
    def test_dif_is_the_experiments_dif_seeded(self, pu_data):
        dif = pytest.importorskip("pyod.models.dif")
        X, y, s = pu_data
        model = IFPNU(prior=y.mean(), detector="DIF", n_estimators=5, random_state=0).fit(X, s)
        detector = model.detector_
        assert isinstance(detector, dif.DIF)
        assert (detector.hidden_activation, detector.contamination) == ("tanh", 0.01)
        assert detector.random_state == 0
        # The experiments ranked the samples by decision_function on the training data.
        from pypu.models._torch import single_thread  # PyTorch next to LightGBM on macOS

        with single_thread():
            scores = detector.decision_function(X)
        np.testing.assert_array_equal(model.pnu_labels_, _pnu_labels(s, scores, y.mean()))
        again = IFPNU(prior=y.mean(), detector="dif", n_estimators=5, random_state=0).fit(X, s)
        np.testing.assert_array_equal(again.pnu_labels_, model.pnu_labels_)
        np.testing.assert_array_equal(again.predict_proba(X), model.predict_proba(X))

    def test_dif_without_pyod_names_the_extra(self, pu_data, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyod.models.dif", None)
        X, y, s = pu_data
        with pytest.raises(ImportError, match=r"pypu-learn\[pyod\]"):
            IFPNU(prior=y.mean(), detector="DIF").fit(X, s)

    def test_rejects_unknown_names(self, pu_data):
        X, y, s = pu_data
        with pytest.raises(ValueError, match="one of 'IF', 'EIF', 'OIF', 'LOF', 'DIF' or an"):
            IFPNU(prior=y.mean(), detector="ECOD").fit(X, s)


def test_detector_is_cloned(pu_data):
    X, y, s = pu_data
    detector = IsolationForest(random_state=0)
    model = IFPNU(prior=y.mean(), detector=detector, n_estimators=5).fit(X, s)
    assert model.detector_ is not detector
    assert not hasattr(detector, "estimators_")


class TestObjective:
    @staticmethod
    def risk(raw, pnu, gamma, prior, eta):
        """Per-sample loss whose derivatives the objective returns."""
        p = expit(raw)
        loss_pos = -((1 - p) ** gamma) * np.log(p)
        loss_neg = -(p**gamma) * np.log(1 - p)
        weight_neg = np.where(pnu == 0, eta * (1 - prior), 1 - eta)
        return np.where(pnu == 1, prior * loss_pos, weight_neg * loss_neg)

    @pytest.mark.parametrize(("gamma", "eta"), [(0.0, 0.5), (1.0, 0.8), (3.0, 0.5), (3.0, 0.2)])
    def test_gradient_and_hessian_match_finite_differences(self, gamma, eta):
        raw = np.linspace(-4, 4, 41).repeat(3)
        pnu = np.tile([1, 0, -1], 41)
        prior, h = 0.3, 1e-4
        objective = _FocalPNUObjective(pnu, prior=prior, eta=eta, gamma=gamma)
        grad, hess = objective(np.zeros_like(raw), raw)
        up, mid, down = (self.risk(raw + d, pnu, gamma, prior, eta) for d in (h, 0, -h))
        np.testing.assert_allclose(grad, (up - down) / (2 * h), atol=1e-6)
        curvature = (up - 2 * mid + down) / h**2
        keep = curvature > 1e-3  # the objective floors the hessian at 1e-6
        np.testing.assert_allclose(hess[keep], curvature[keep], rtol=1e-3, atol=1e-5)
        assert np.all(hess >= 1e-6)

    def test_gamma_zero_is_weighted_cross_entropy(self):
        raw = np.linspace(-3, 3, 12)
        pnu = np.tile([1, 0, -1], 4)
        p = expit(raw)
        grad, _ = _FocalPNUObjective(pnu, prior=0.2, eta=0.75, gamma=0.0)(np.zeros(12), raw)
        weight = np.select([pnu == 1, pnu == 0], [0.2, 0.75 * 0.8], 0.25)
        target = (pnu == 1).astype(float)
        np.testing.assert_allclose(grad, weight * (p - target), atol=1e-12)

    def test_rejects_another_training_set(self):
        objective = _FocalPNUObjective(np.array([1, 0, -1]), prior=0.3, eta=0.5, gamma=3.0)
        with pytest.raises(ValueError, match="another training set"):
            objective(np.zeros(4), np.zeros(4))


@pytest.mark.parametrize("eta", [-0.1, 1.5, float("nan"), "0.5", True])
def test_rejects_invalid_eta(pu_data, eta):
    X, y, s = pu_data
    with pytest.raises(ValueError, match="eta must be a number in"):
        IFPNU(prior=y.mean(), eta=eta).fit(X, s)


def test_requires_the_prior(pu_data):
    X, _, s = pu_data
    with pytest.raises(ValueError, match="IFPNU needs the class prior"):
        IFPNU().fit(X, s)


@pytest.mark.parametrize("gamma", [-1, float("inf"), True])
def test_rejects_invalid_gamma(pu_data, gamma):
    X, y, s = pu_data
    with pytest.raises(ValueError, match="gamma must be"):
        IFPNU(prior=y.mean(), gamma=gamma).fit(X, s)


@pytest.mark.parametrize("key", ["objective", "num_boost_round"])
def test_lgbm_params_cannot_set_the_objective_or_rounds(pu_data, key):
    X, y, s = pu_data
    with pytest.raises(ValueError, match="lgbm_params cannot set"):
        IFPNU(prior=y.mean(), lgbm_params={key: 5}).fit(X, s)
