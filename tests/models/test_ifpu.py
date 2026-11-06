from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from pypu.models import IFPU
from pypu.models.ifpu import _FocalPUObjective

REFERENCE = Path(__file__).with_name("data") / "ifpu_reference.npz"

# Reference cases of data/make_ifpu_reference.py -> the IFPU parameters that match them.
REFERENCE_CASES = {
    "published_scar": {},
    "published_sar": {},
    "gamma1_50trees": {"gamma": 1, "learning_rate": 0.05, "n_estimators": 50},
}


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


@pytest.mark.parametrize("case", list(REFERENCE_CASES))
def test_reproduces_the_original_implementation(reference, case):
    model = IFPU(prior=float(reference["prior"]), **REFERENCE_CASES[case])
    model.fit(reference["X_train"], reference[f"{case}_s"])
    assert model.booster_.num_trees() == reference[f"{case}_n_trees"]
    np.testing.assert_allclose(
        model.predict_proba(reference["X_test"])[:, 1], reference[f"{case}_scores"], atol=1e-9
    )


def test_defaults_are_the_published_settings(pu_data):
    X, y, s = pu_data
    model = IFPU(prior=y.mean()).fit(X, s)
    assert model.booster_.num_trees() == 10
    assert model.booster_.params["learning_rate"] == 0.1
    assert model.gamma == 3.0


def test_decision_function_is_the_raw_score(pu_data):
    X, y, s = pu_data
    model = IFPU(prior=y.mean(), n_estimators=20).fit(X, s)
    raw = model.decision_function(X)
    np.testing.assert_allclose(raw, model.booster_.predict(X, raw_score=True))
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], expit(raw))
    np.testing.assert_array_equal(model.predict(X), (raw >= 0).astype(int))


def test_lgbm_params_are_passed_on(pu_data):
    X, y, s = pu_data
    model = IFPU(
        prior=y.mean(), random_state=0, lgbm_params={"subsample": 0.5, "subsample_freq": 1}
    ).fit(X, s)
    assert model.booster_.params["subsample"] == 0.5
    full = IFPU(prior=y.mean(), random_state=0).fit(X, s)
    assert not np.allclose(model.decision_function(X), full.decision_function(X))


@pytest.mark.parametrize("key", ["objective", "num_boost_round", "n_estimators", "num_trees"])
def test_lgbm_params_cannot_set_the_objective_or_rounds(pu_data, key):
    X, y, s = pu_data
    with pytest.raises(ValueError, match="lgbm_params cannot set"):
        IFPU(prior=y.mean(), lgbm_params={key: 5}).fit(X, s)


def test_requires_the_prior(pu_data):
    X, _, s = pu_data
    with pytest.raises(ValueError, match="IFPU needs the class prior"):
        IFPU().fit(X, s)


@pytest.mark.parametrize(
    ("prior", "error"), [(1.5, ValueError), (0, ValueError), ("0.3", TypeError)]
)
def test_rejects_invalid_prior(pu_data, prior, error):
    X, _, s = pu_data
    with pytest.raises(error, match="prior must be"):
        IFPU(prior=prior).fit(X, s)


@pytest.mark.parametrize("gamma", [-1, float("nan"), float("inf"), "3", True])
def test_rejects_invalid_gamma(pu_data, gamma):
    X, y, s = pu_data
    with pytest.raises(ValueError, match="gamma must be"):
        IFPU(prior=y.mean(), gamma=gamma).fit(X, s)


class TestObjective:
    @staticmethod
    def risk(raw, y, gamma, prior):
        """Per-sample loss whose derivatives the objective returns."""
        p = expit(raw)
        loss_pos = -((1 - p) ** gamma) * np.log(p)
        loss_neg = -(p**gamma) * np.log(1 - p)
        return y * loss_pos + (1 - y) * (prior * loss_pos + (1 - prior) * loss_neg)

    @pytest.mark.parametrize("gamma", [0.0, 1.0, 3.0])
    def test_gradient_and_hessian_match_finite_differences(self, gamma):
        raw = np.linspace(-4, 4, 41).repeat(2)
        y = np.tile([1.0, 0.0], 41)
        prior, h = 0.3, 1e-4
        grad, hess = _FocalPUObjective(gamma=gamma, prior=prior)(y, raw)
        up, mid, down = (self.risk(raw + d, y, gamma, prior) for d in (h, 0, -h))
        np.testing.assert_allclose(grad, (up - down) / (2 * h), atol=1e-6)
        curvature = (up - 2 * mid + down) / h**2
        keep = curvature > 1e-3  # the objective floors the hessian at 1e-6
        np.testing.assert_allclose(hess[keep], curvature[keep], rtol=1e-3, atol=1e-5)
        assert np.all(hess >= 1e-6)

    def test_gamma_zero_is_cross_entropy_with_soft_labels(self):
        raw = np.linspace(-3, 3, 13)
        y = (np.arange(13) % 2).astype(float)
        p = expit(raw)
        grad, hess = _FocalPUObjective(gamma=0.0, prior=0.3)(y, raw)
        # Unlabeled samples are fitted towards the prior, labeled positives towards 1.
        np.testing.assert_allclose(grad, np.where(y == 1, p - 1, p - 0.3), atol=1e-12)
        np.testing.assert_allclose(hess, p * (1 - p), atol=1e-12)
