from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, log_expit

torch = pytest.importorskip("torch")
pytest.importorskip("cvxopt")

from pypu.models import PUe  # noqa: E402
from pypu.models._torch import single_thread  # noqa: E402
from pypu.models.pue import _classifier_loss, _propensity_loss  # noqa: E402

REFERENCE = Path(__file__).with_name("data") / "pue_reference.npz"
CASES = ["scar", "sar"]


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


@pytest.mark.parametrize("case", CASES)
def test_reproduces_the_original(reference, case):
    # The original's KM prior and 20 Adam steps per model: under PyTorch 2.8 (the reference)
    # and 2.15 alike, PyPU gives the same probabilities and weights to the last bit. Other
    # platforms' float32 arithmetic can differ in the last bits.
    model = PUe(random_state=int(reference[f"{case}_seed"]))
    model.fit(reference["X_train"], reference[f"{case}_s"])
    assert model.prior_ == pytest.approx(float(reference[f"{case}_prior"]), rel=1e-12)
    np.testing.assert_allclose(
        model.predict_proba(reference["X_test"]), reference[f"{case}_proba"], atol=1e-5
    )
    for attribute, key in [
        ("coef_", "clf_weight"),
        ("intercept_", "clf_bias"),
        ("propensity_coef_", "e_weight"),
        ("propensity_intercept_", "e_bias"),
    ]:
        np.testing.assert_allclose(
            np.ravel(getattr(model, attribute)), reference[f"{case}_{key}"].ravel(), atol=1e-5
        )


def test_initial_weights_are_those_of_torch_manual_seed(pu_data):
    # One step with a learning rate of ~0 leaves the weights at their initial values: those
    # of two nn.Linear layers created after torch.manual_seed, the propensity model first.
    X, _, s = pu_data
    model = PUe(prior=0.3, n_steps=1, learning_rate=1e-30, random_state=5).fit(X, s)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        propensity = torch.nn.Linear(X.shape[1], 1)
        classifier = torch.nn.Linear(X.shape[1], 1)
    np.testing.assert_allclose(model.propensity_coef_, propensity.weight.detach().ravel())
    np.testing.assert_allclose(model.coef_, classifier.weight.detach().ravel())


def test_classifier_loss_is_the_documented_objective():
    rng = np.random.default_rng(0)
    logit = rng.normal(size=(50, 1))
    s = (rng.random((50, 1)) < 0.3).astype(float)
    normalized = rng.uniform(0.5, 2.0, size=(50, 1))  # sum over labeled of 1/e, times e
    prior = 0.3
    weights = s / normalized
    expected = (
        prior * np.sum(weights * -log_expit(logit))
        + np.sum((1 - s) * -log_expit(-logit))
        - prior * np.sum(weights * -log_expit(-logit))
    )
    with single_thread():
        loss = _classifier_loss(
            torch.from_numpy(logit), torch.from_numpy(s), torch.from_numpy(normalized), prior
        )
    assert loss.item() == pytest.approx(expected)


def test_propensity_loss_is_the_documented_objective():
    rng = np.random.default_rng(1)
    logit = rng.normal(size=(40, 1))
    s = (rng.random((40, 1)) < 0.3).astype(float)
    n_labeled = s.sum()
    expected = np.mean(-s * log_expit(logit) - (1 - s) * log_expit(-logit)) + 15.0 * abs(
        expit(logit).sum() - n_labeled
    )
    s_t = torch.from_numpy(s)
    with single_thread():
        loss = _propensity_loss(
            torch.from_numpy(logit), s_t, torch.sum(s_t), 40 - torch.sum(s_t), 15.0
        )
    assert loss.item() == pytest.approx(expected)


def test_longer_training_matches_the_number_of_labeled_samples(pu_data):
    # The penalty pulls the sum of the propensities towards the number of labeled samples.
    X, _, s = pu_data
    model = PUe(prior=0.3, n_steps=2000, learning_rate=0.01, random_state=0).fit(X, s)
    propensity = expit(X @ model.propensity_coef_ + model.propensity_intercept_)
    assert propensity.sum() == pytest.approx(s.sum(), rel=0.02)


def test_given_prior_skips_the_estimate(pu_data):
    X, _, s = pu_data
    assert PUe(prior=0.3, random_state=0).fit(X, s).prior_ == 0.3


def test_estimated_prior_is_km1(reference):
    from pypu.prior import KMPrior

    X, s = reference["X_train"], reference["scar_s"]
    model = PUe(random_state=0).fit(X, s)
    # PUe combines KM1 with a float32 mean of s, as the original did (2e-8 apart here).
    assert model.prior_ == pytest.approx(KMPrior().fit(X, s).prior_, rel=1e-6)


def test_decision_function_is_the_logit(pu_data):
    X, _, s = pu_data
    model = PUe(prior=0.3, random_state=0).fit(X, s)
    logit = model.decision_function(X)
    np.testing.assert_allclose(logit, X @ model.coef_ + model.intercept_, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], expit(logit), rtol=1e-6)


def test_random_state(pu_data):
    X, _, s = pu_data
    first = PUe(prior=0.3, random_state=0).fit(X, s).coef_
    np.testing.assert_array_equal(PUe(prior=0.3, random_state=0).fit(X, s).coef_, first)
    assert not np.array_equal(PUe(prior=0.3, random_state=1).fit(X, s).coef_, first)


def test_defaults_are_the_experiments_settings():
    params = PUe().get_params()
    assert params["prior"] is None
    assert params["n_steps"] == 20
    assert params["learning_rate"] == 0.001
    assert params["alpha"] == 15.0


def test_divergence_raises(pu_data):
    X, _, s = pu_data
    with pytest.raises(ValueError, match="PUe diverged"):
        PUe(prior=0.3, random_state=0).fit(X * 1e30, s)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"n_steps": 0}, "n_steps must be a positive int"),
        ({"learning_rate": -1.0}, "learning_rate must be a positive number"),
        ({"alpha": -1.0}, "alpha must be a number >= 0"),
        ({"alpha": "15"}, "alpha must be a number >= 0"),
        ({"prior": 1.5}, "prior must be"),
    ],
)
def test_rejects_invalid_parameters(pu_data, params, match):
    X, _, s = pu_data
    with pytest.raises(ValueError, match=match):
        PUe(**{"prior": 0.3, **params}).fit(X, s)
