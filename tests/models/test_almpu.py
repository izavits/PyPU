from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

torch = pytest.importorskip("torch")
pytest.importorskip("jenkspy")

from pypu.models import ALMPU  # noqa: E402
from pypu.models.almpu import _cosine_schedule  # noqa: E402

REFERENCE = Path(__file__).with_name("data") / "almpu_reference.npz"
SETTINGS = {
    "experiments": {},
    "short": {"n_warmup_epochs": 3, "n_finetune_epochs": 4, "n_steps_per_epoch": 10},
    "short_no_ema": {
        "n_warmup_epochs": 3,
        "n_finetune_epochs": 3,
        "n_steps_per_epoch": 10,
        "ema_decay": None,
        "batch_size": 32,
    },
}
SHORT = SETTINGS["short"]


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def fitted(reference):
    return {
        case: ALMPU(random_state=int(reference[f"{case}_seed"]), **settings).fit(
            reference["X_train"], reference[f"{case}_s"]
        )
        for case, settings in SETTINGS.items()
    }


# Under the same PyTorch, PyPU and the original train the same networks to the last bit. The
# reference comes from PyTorch 2.14, the experiments' version. SGD amplifies the rounding
# differences between versions: over the experiments' 5000 steps they grow into different
# networks (under PyTorch 2.15 the original's own test scores differ from 2.14's by up to
# 0.99), while short trainings stay close.


@pytest.mark.parametrize("case", ["short", "short_no_ema"])
def test_short_training_reproduces_the_original(reference, fitted, case):
    model = fitted[case]
    expected = (reference[f"{case}_pseudo_targets"] == 0).astype(int)  # original: 0 = positive
    np.testing.assert_array_equal(model.pseudo_labels_, expected)
    np.testing.assert_allclose(model.trends_, reference[f"{case}_trends"], rtol=1e-4, atol=1e-7)
    assert model.break_point_ == pytest.approx(float(reference[f"{case}_intervals"][1]), rel=1e-4)
    assert model.pollution_ == pytest.approx(expected.mean())
    proba = model.predict_proba(reference["X_test"])[:, 1]
    np.testing.assert_allclose(proba, reference[f"{case}_scores"], atol=1e-4)


def test_experiments_setting_gives_the_original_pseudo_labels(reference, fitted):
    # Phase 1's 2000 steps move the trends by up to 0.01 across PyTorch versions and platforms,
    # which can flip the samples nearest the break: all pseudo-labels agree under PyTorch 2.15
    # on macOS, 4 of 367 differ on Linux. So this only checks that the default pipeline gives
    # nearly the same pseudo-labels.
    model = fitted["experiments"]
    expected = (reference["experiments_pseudo_targets"] == 0).astype(int)
    assert np.mean(model.pseudo_labels_ == expected) >= 0.95
    assert model.pollution_ == pytest.approx(expected.mean(), abs=0.03)


def test_fit_leaves_the_global_torch_state_alone(pu_data):
    X, _, s = pu_data
    state = torch.random.get_rng_state()
    ALMPU(random_state=0, **SHORT).fit(X, s)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_random_state(pu_data):
    X, _, s = pu_data
    first = ALMPU(random_state=0, **SHORT).fit(X, s).decision_function(X)
    np.testing.assert_array_equal(
        ALMPU(random_state=0, **SHORT).fit(X, s).decision_function(X), first
    )
    assert not np.array_equal(ALMPU(random_state=1, **SHORT).fit(X, s).decision_function(X), first)


def test_attributes(pu_data):
    X, _, s = pu_data
    model = ALMPU(random_state=0, **SHORT).fit(X, s)
    n_unlabeled = int((s == 0).sum())
    assert model.pseudo_labels_.shape == model.trends_.shape == (n_unlabeled,)
    np.testing.assert_array_equal(model.pseudo_labels_, model.trends_ > model.break_point_)
    assert model.pollution_ == pytest.approx(model.pseudo_labels_.mean())
    assert isinstance(model.network_, torch.nn.Module)
    assert not model.network_.training


def test_decision_function_is_the_output_difference(pu_data):
    X, _, s = pu_data
    model = ALMPU(random_state=0, **SHORT).fit(X, s)
    logit = model.decision_function(X)
    with torch.no_grad():
        outputs = model.network_.double()(torch.from_numpy(X)).numpy()
    np.testing.assert_allclose(logit, outputs[:, 0] - outputs[:, 1], rtol=1e-12)
    np.testing.assert_array_equal(model.predict_proba(X)[:, 1], expit(logit))


def test_predictions_do_not_depend_on_batching(pu_data):
    X, _, s = pu_data
    model = ALMPU(random_state=0, **SHORT).fit(X, s)
    one_by_one = np.concatenate([model.decision_function(X[i : i + 1]) for i in range(len(X))])
    np.testing.assert_allclose(model.decision_function(X), one_by_one, rtol=1e-12, atol=1e-12)


def test_cosine_schedule():
    factor = _cosine_schedule(100)
    assert factor(0) == 1.0
    assert factor(100) == pytest.approx(np.cos(np.pi * 7 / 16))
    assert all(factor(step) >= factor(step + 1) for step in range(100))


def test_defaults_are_the_experiments_settings():
    params = ALMPU().get_params()
    assert params["batch_size"] == 64
    assert params["n_steps_per_epoch"] == 100
    assert params["n_warmup_epochs"] == 20
    assert params["n_finetune_epochs"] == 30
    assert params["learning_rate"] == 0.01
    assert params["weight_decay"] == 5e-4
    assert params["ema_decay"] == 0.99
    assert params["label_smoothing"] == 0.1
    assert params["alpha"] == 7e-5
    assert params["false_alarm_cutoff"] == 0.05
    assert params["penalty_multiplier"] == 1.05


def test_identical_trends_raise(pu_data):
    X, _, s = pu_data
    with pytest.raises(ValueError, match="cannot split the unlabeled samples"):
        ALMPU(random_state=0, **SHORT).fit(np.zeros_like(X), s)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"n_warmup_epochs": 1}, "n_warmup_epochs must be at least 2"),
        ({"n_finetune_epochs": 0}, "n_finetune_epochs must be a positive int"),
        ({"batch_size": 0}, "batch_size must be a positive int"),
        ({"learning_rate": 0.0}, "learning_rate must be a positive number"),
        ({"ema_decay": 1.0}, "ema_decay must be None or a number in"),
        ({"label_smoothing": 1.0}, "label_smoothing must be a number in"),
        ({"weight_decay": -1.0}, "weight_decay must be a finite number >= 0"),
        ({"penalty_multiplier": 0.0}, "penalty_multiplier must be a positive number"),
    ],
)
def test_rejects_invalid_parameters(pu_data, params, match):
    X, _, s = pu_data
    with pytest.raises(ValueError, match=match):
        ALMPU(**params).fit(X, s)
