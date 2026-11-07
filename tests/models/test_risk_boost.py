import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import roc_auc_score

xgb = pytest.importorskip("xgboost")

from pypu.models import ImbalancedNNPUBoost, NNPUBoost, UPUBoost  # noqa: E402
from pypu.models.risk_boost import (  # noqa: E402
    _ImbalancedNNPUObjective,
    _NNPUObjective,
    _UPUObjective,
)

REFERENCE = Path(__file__).with_name("data") / "risk_boost_reference.npz"


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


def objectives(reference):
    prior = float(reference["prior"])
    c_inverse = prior / float(reference["s_train"].mean())
    return {
        "nnpu_experiments": _NNPUObjective(c_inverse),
        "upu_experiments": _UPUObjective(c_inverse),
        "imbnnpu_experiments": _ImbalancedNNPUObjective(prior, 0.5),
        "imbnnpu_oversampling03": _ImbalancedNNPUObjective(prior, 0.3),
    }


def models(prior):
    return {
        "nnpu_experiments": NNPUBoost(prior=prior, random_state=0),
        "upu_experiments": UPUBoost(prior=prior, random_state=0),
        "imbnnpu_experiments": ImbalancedNNPUBoost(prior=prior, random_state=0),
        "imbnnpu_oversampling03": ImbalancedNNPUBoost(
            prior=prior, oversampled_prior=0.3, random_state=1
        ),
    }


@pytest.mark.parametrize(
    "case",
    ["nnpu_experiments", "upu_experiments", "imbnnpu_experiments", "imbnnpu_oversampling03"],
)
def test_objectives_match_the_original(reference, case):
    # The original ran under NumPy 1, so most of its arithmetic was float32. On the original's
    # platform the gradients agree bit for bit; across platforms float32 sigmoids can differ in
    # the last bit or two.
    dtrain = xgb.DMatrix(reference["X_train"], label=reference["s_train"])
    grad, hess = objectives(reference)[case](reference["raw_grid"], dtrain)
    assert grad.dtype == reference[f"{case}_grad"].dtype
    np.testing.assert_allclose(grad, reference[f"{case}_grad"], rtol=3e-7, atol=1e-12)
    np.testing.assert_allclose(hess, reference[f"{case}_hess"], rtol=3e-7, atol=1e-12)


@pytest.mark.parametrize(
    "case", ["nnpu_experiments", "imbnnpu_experiments", "imbnnpu_oversampling03"]
)
def test_reproduces_the_original_scores(reference, case):
    # Exact under the original's XGBoost 2.1 and platform; about 1e-6 across platforms and
    # 1e-3 under XGBoost 3, whose tree construction changed slightly.
    model = models(float(reference["prior"]))[case].fit(reference["X_train"], reference["s_train"])
    assert model.booster_.num_boosted_rounds() == reference[f"{case}_n_rounds"]
    np.testing.assert_allclose(
        model.decision_function(reference["X_test"]), reference[f"{case}_raw"], rtol=0, atol=1e-2
    )


def test_upu_ranks_like_the_original(reference):
    # uPU's risk is unbounded below, so its scores grow large (about 100 here) and amplify the
    # last-bit differences above; the ranking of the test samples stays the same.
    model = UPUBoost(prior=float(reference["prior"]), random_state=0)
    raw = model.fit(reference["X_train"], reference["s_train"]).decision_function(
        reference["X_test"]
    )
    y_test = reference["y_test"]
    assert roc_auc_score(y_test, raw) == pytest.approx(
        roc_auc_score(y_test, reference["upu_experiments_raw"]), abs=1e-3
    )


class TestObjectives:
    @pytest.fixture
    def setup(self):
        raw = np.linspace(-4, 4, 40, dtype=np.float32).repeat(2)
        s = np.tile([1.0, 0.0], 40)
        return raw, s, xgb.DMatrix(np.zeros((80, 1)), label=s)

    @staticmethod
    def cross_entropies(raw):
        p = expit(raw.astype(np.float64))
        return -np.log(p), -np.log(1 - p)

    def check_derivatives(self, objective, risk, raw, dtrain):
        h = 1e-3
        grad, hess = objective(raw, dtrain)
        raw64 = raw.astype(np.float64)
        up, mid, down = (risk(raw64 + d) for d in (h, 0, -h))
        np.testing.assert_allclose(grad, (up - down) / (2 * h), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(hess, (up - 2 * mid + down) / h**2, rtol=1e-3, atol=1e-4)

    def test_upu_is_the_unbiased_risk(self, setup):
        raw, s, dtrain = setup
        c_inverse = 2.5

        def risk(z):
            positive, negative = self.cross_entropies(z)
            return np.where(s == 1, c_inverse * positive + (1 - c_inverse) * negative, negative)

        self.check_derivatives(_UPUObjective(c_inverse), risk, raw, dtrain)

    def test_nnpu_is_a_weighted_cross_entropy(self, setup):
        raw, s, dtrain = setup
        c_inverse = 2.5

        def risk(z):
            positive, negative = self.cross_entropies(z)
            return np.where(s == 1, c_inverse * positive, negative)

        self.check_derivatives(_NNPUObjective(c_inverse), risk, raw, dtrain)

    def test_imbalanced_nnpu_is_a_weighted_cross_entropy(self, setup):
        raw, s, dtrain = setup
        prior, rho = 0.1, 0.5

        def risk(z):
            positive, negative = self.cross_entropies(z)
            return np.where(s == 1, rho * positive, (1 - rho) / (1 - prior) * negative)

        self.check_derivatives(_ImbalancedNNPUObjective(prior, rho), risk, raw, dtrain)


@pytest.mark.parametrize("model", [UPUBoost, NNPUBoost, ImbalancedNNPUBoost])
def test_defaults_are_the_experiments_settings(pu_data, model):
    X, y, s = pu_data
    clf = model(prior=y.mean(), random_state=0).fit(X, s)
    config = json.loads(clf.booster_.save_config())["learner"]["gradient_booster"]
    tree_params = config["tree_train_param"]
    assert clf.booster_.num_boosted_rounds() == 100
    assert float(tree_params["eta"]) == pytest.approx(0.3)
    assert int(tree_params["max_depth"]) == 6
    assert float(tree_params["reg_lambda"]) == 0.0
    assert config["gbtree_train_param"]["tree_method"] == "exact"


@pytest.mark.parametrize("model", [UPUBoost, NNPUBoost, ImbalancedNNPUBoost])
def test_decision_function_is_the_raw_score(pu_data, model):
    X, y, s = pu_data
    clf = model(prior=y.mean(), n_estimators=10, random_state=0).fit(X, s)
    raw = clf.decision_function(X)
    np.testing.assert_allclose(clf.predict_proba(X)[:, 1], expit(raw), rtol=1e-6)
    np.testing.assert_array_equal(clf.predict(X), (expit(raw) > 0.5).astype(int))


def test_xgb_params_are_passed_on(pu_data):
    X, y, s = pu_data
    clf = NNPUBoost(prior=y.mean(), random_state=0, xgb_params={"max_bin": 32}).fit(X, s)
    assert "max_bin" in clf.booster_.save_config()


@pytest.mark.parametrize("key", ["objective", "obj"])
def test_xgb_params_cannot_set_the_objective(pu_data, key):
    X, y, s = pu_data
    with pytest.raises(ValueError, match="xgb_params cannot set the objective"):
        NNPUBoost(prior=y.mean(), xgb_params={key: "binary:logistic"}).fit(X, s)


@pytest.mark.parametrize("model", [UPUBoost, NNPUBoost, ImbalancedNNPUBoost])
def test_requires_the_prior(pu_data, model):
    X, _, s = pu_data
    with pytest.raises(ValueError, match="needs the class prior"):
        model().fit(X, s)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"n_estimators": 0}, "n_estimators must be a positive int"),
        ({"random_state": np.random.RandomState(0)}, "random_state must be an int or None"),
    ],
)
def test_rejects_invalid_parameters(pu_data, params, match):
    X, y, s = pu_data
    with pytest.raises(ValueError, match=match):
        UPUBoost(prior=y.mean(), **params).fit(X, s)


@pytest.mark.parametrize("oversampled_prior", [0.0, 1.0, "0.5"])
def test_rejects_invalid_oversampled_prior(pu_data, oversampled_prior):
    X, y, s = pu_data
    with pytest.raises((ValueError, TypeError), match="prior must be"):
        ImbalancedNNPUBoost(prior=y.mean(), oversampled_prior=oversampled_prior).fit(X, s)


def test_missing_xgboost_gives_an_install_hint():
    code = (
        "import sys; sys.modules['xgboost'] = None\n"
        "import pypu.models\n"
        "try:\n"
        "    pypu.models.NNPUBoost\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert "pip install 'pypu-learn[xgboost]'" in output
