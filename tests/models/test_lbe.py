import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

torch = pytest.importorskip("torch")

from pypu.models import LBE  # noqa: E402
from pypu.models._torch import init_linear_, make_generator, single_thread  # noqa: E402
from pypu.models.lbe import _e_step, _Logistic, _negative_log_likelihood  # noqa: E402

REFERENCE = Path(__file__).with_name("data") / "lbe_reference.npz"
CASES = ["scar", "sar"]
SHORT = {"n_iter": 5, "n_m_steps": 10, "n_pretrain_steps": 50}


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def fitted(reference):
    """PyPU's LBE fitted as in the reference, once per case (each fit takes a few seconds)."""
    return {
        case: LBE(random_state=int(reference[f"{case}_seed"])).fit(
            reference["X_train"], reference[f"{case}_s"]
        )
        for case in CASES
    }


def logistic(weight, bias):
    model = _Logistic(weight.shape[1])
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(weight))
        model.bias.copy_(torch.from_numpy(bias))
    return model


@pytest.mark.parametrize("case", CASES)
def test_e_step_and_gradients_match_the_original(reference, case):
    # At the original's fitted weights, one E-step and the M-step loss and gradients. These
    # do not accumulate rounding over a fit, so they agree closely on any platform. Like LBE
    # itself, run PyTorch on one thread: on macOS, once LightGBM has run in the process,
    # PyTorch's own OpenMP threads deadlock.
    with single_thread():
        check_e_step_and_gradients(reference, case)


def check_e_step_and_gradients(reference, case):
    classifier = logistic(reference[f"{case}_h_weight"], reference[f"{case}_h_bias"])
    propensity = logistic(reference[f"{case}_eta_weight"], reference[f"{case}_eta_bias"])
    X = torch.from_numpy(reference["X_train"])
    s = torch.from_numpy(reference[f"{case}_s"]).float()

    # Absolute tolerance: float32 sigmoids differ in the last bit or two across platforms,
    # and 1 - h(x) turns that into a large relative difference when h(x) is close to 1
    # (8e-4 for one sample on Linux; 1.4e-7 in absolute terms).
    posterior = _e_step(classifier, propensity, X, s)
    np.testing.assert_allclose(posterior.numpy(), reference[f"{case}_posterior"], atol=1e-6)

    loss = _negative_log_likelihood(classifier(X), propensity(X), s, posterior)
    loss.backward()
    assert loss.item() == pytest.approx(float(reference[f"{case}_loss"]), rel=1e-5)
    for name, parameter in [
        ("h_weight", classifier.weight),
        ("h_bias", classifier.bias),
        ("eta_weight", propensity.weight),
        ("eta_bias", propensity.bias),
    ]:
        np.testing.assert_allclose(
            parameter.grad.numpy(), reference[f"{case}_grad_{name}"], rtol=1e-4, atol=1e-5
        )


@pytest.mark.parametrize("case", CASES)
def test_reproduces_the_original(reference, fitted, case):
    # Under the same PyTorch, PyPU and the original give the same probabilities to the last
    # bit. The reference comes from PyTorch 2.8, the experiments' version; 11,000 float32 Adam
    # steps amplify the small numerical changes of later versions (1.5e-4 under PyTorch 2.15,
    # for the original code too).
    proba = fitted[case].predict_proba(reference["X_test"])
    np.testing.assert_allclose(proba, reference[f"{case}_proba"], atol=2e-3)
    np.testing.assert_allclose(
        fitted[case].coef_, reference[f"{case}_h_weight"].ravel(), rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize("case", CASES)
def test_propensity_is_a_classifier_of_the_labels(reference, fitted, case):
    # As run, the propensity model's terms add up to a logistic regression of s on x.
    X, s = reference["X_train"], reference[f"{case}_s"]
    model = fitted[case]
    eta_logit = X @ model.propensity_coef_ + model.propensity_intercept_
    plain = LogisticRegression(C=1e6, max_iter=10_000).fit(X, s).decision_function(X)
    assert np.corrcoef(eta_logit, plain)[0, 1] > 0.9999


@pytest.mark.parametrize("seed", [0, 3, 123])
def test_initial_weights_are_those_of_torch_manual_seed(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        linear = torch.nn.Linear(8, 1)
    model = _Logistic(8)
    init_linear_(model.weight, model.bias, make_generator(seed))
    assert torch.equal(model.weight, linear.weight)
    assert torch.equal(model.bias, linear.bias)


def test_fit_leaves_the_global_torch_state_alone(pu_data):
    X, _, s = pu_data
    state = torch.random.get_rng_state()
    LBE(random_state=0, **SHORT).fit(X, s)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_random_state(pu_data):
    X, _, s = pu_data
    first = LBE(random_state=0, **SHORT).fit(X, s).coef_
    np.testing.assert_array_equal(LBE(random_state=0, **SHORT).fit(X, s).coef_, first)
    assert not np.array_equal(LBE(random_state=1, **SHORT).fit(X, s).coef_, first)
    rng = np.random.RandomState(0)
    assert LBE(random_state=rng, **SHORT).fit(X, s).coef_.shape == (X.shape[1],)


def test_decision_function_is_the_logit(pu_data):
    X, _, s = pu_data
    model = LBE(random_state=0, **SHORT).fit(X, s)
    logit = model.decision_function(X)
    np.testing.assert_allclose(logit, X @ model.coef_ + model.intercept_, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], expit(logit), rtol=1e-6)


def test_more_steps_change_the_fit(pu_data):
    X, _, s = pu_data
    short = LBE(random_state=0, **SHORT).fit(X, s).coef_
    longer = LBE(random_state=0, **{**SHORT, "n_iter": 6}).fit(X, s).coef_
    assert not np.array_equal(short, longer)


def test_defaults_are_the_experiments_settings():
    params = LBE().get_params()
    assert params["n_iter"] == 100
    assert params["n_m_steps"] == 100
    assert params["n_pretrain_steps"] == 1000
    assert params["learning_rate"] == 0.01


def test_accepts_a_torch_device(pu_data):
    X, _, s = pu_data
    model = LBE(device=torch.device("cpu"), random_state=0, **SHORT).fit(X, s)
    assert model.predict(X).shape == (X.shape[0],)


def test_divergence_raises(pu_data):
    X, _, s = pu_data
    with pytest.raises(ValueError, match="LBE diverged"):
        LBE(random_state=0, **SHORT).fit(X * 1e30, s)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"n_iter": 0}, "n_iter must be a positive int"),
        ({"n_m_steps": 2.5}, "n_m_steps must be a positive int"),
        ({"n_pretrain_steps": True}, "n_pretrain_steps must be a positive int"),
        ({"learning_rate": 0.0}, "learning_rate must be a positive number"),
        ({"learning_rate": float("nan")}, "learning_rate must be a positive number"),
    ],
)
def test_rejects_invalid_parameters(pu_data, params, match):
    X, _, s = pu_data
    with pytest.raises(ValueError, match=match):
        LBE(**params).fit(X, s)


def test_missing_torch_gives_an_install_hint():
    # A finder that hides PyTorch (sys.modules['torch'] = None would trip SciPy's array API
    # helpers, which look torch up in sys.modules).
    code = (
        "import sys\n"
        "class HideTorch:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'torch':\n"
        "            raise ModuleNotFoundError(name)\n"
        "sys.meta_path.insert(0, HideTorch())\n"
        "import pypu.models\n"
        "try:\n"
        "    pypu.models.LBE\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert "pip install 'pypu-learn[torch]'" in output
